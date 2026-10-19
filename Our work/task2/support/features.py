import json
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from prepare import digest, save_json

HERE = Path(__file__).resolve().parent

STAT_NAMES = ['log_chars', 'log_space_words', 'log_newlines', 'digit_fraction',
              'math_symbol_fraction', 'code_keyword', 'question_fraction', 'cjk_fraction']


def statistics(texts):
    rows = []
    for text in texts:
        n = max(1, len(text))
        rows.append([np.log1p(len(text)), np.log1p(len(text.split())), np.log1p(text.count('\n')),
                     sum(c.isdigit() for c in text)/n,
                     sum(c in '=+-*/^∑√<>∫' for c in text)/n,
                     float(bool(re.search(r'\b(def|class|return|import|function|print)\b|```', text))),
                     (text.count('?') + text.count('？'))/n,
                     sum('\u4e00' <= c <= '\u9fff' for c in text)/n])
    return np.asarray(rows, dtype=np.float32)


def local_minilm_path():
    root = Path.home() / '.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'
    candidates = sorted((root / 'snapshots').glob('*'))
    valid = [p for p in candidates if (p / 'model.safetensors').exists() and (p / 'tokenizer.json').exists()]
    if not valid:
        raise FileNotFoundError('MiniLM snapshot not cached. Supply a local model path or explicitly use --encoder tfidf_svd.')
    return valid[-1]


def load_encoder(path, threads):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['USE_TF'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(threads)
    return SentenceTransformer(str(path), device='cpu', local_files_only=True)


def chunk_embeddings(texts, model, batch_size):
    import torch
    tokenizer = model.tokenizer
    limit = int(model.max_seq_length)
    payload = limit - tokenizer.num_special_tokens_to_add(pair=False)
    token_lists = tokenizer(list(texts), add_special_tokens=False, truncation=False)['input_ids']
    unique, lookup, mapping, lengths = [], {}, [], []
    for tokens in token_lists:
        chunks = [tuple(tokens[j:j+payload]) for j in range(0, len(tokens), payload)] or [tuple()]
        row = []
        for chunk in chunks:
            if chunk not in lookup:
                lookup[chunk] = len(unique)
                unique.append(chunk)
            row.append(lookup[chunk])
        mapping.append(row)
        lengths.append(len(tokens))
    embeddings = np.empty((len(unique), model.get_sentence_embedding_dimension()), dtype=np.float32)
    max_encoded_length = 0
    started = time.perf_counter()
    model.eval()
    for p in range(0, len(unique), batch_size):
        records = [tokenizer.prepare_for_model(list(ids), add_special_tokens=True, truncation=False,
                    return_attention_mask=True) for ids in unique[p:p+batch_size]]
        batch = tokenizer.pad(records, padding=True, return_tensors='pt')
        max_encoded_length = max(max_encoded_length, int(batch['input_ids'].shape[1]))
        if max_encoded_length > limit:
            raise AssertionError('Chunk exceeded encoder length; refusing silent truncation')
        with torch.inference_mode():
            emb = model(batch)['sentence_embedding']
            emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()
        embeddings[p:p+len(records)] = emb
        if p == 0 or (p // batch_size + 1) % 100 == 0:
            print('Encoded %d/%d unique chunks (%.1fs)' % (p+len(records), len(unique), time.perf_counter()-started), flush=True)
    result = np.vstack([embeddings[ids].mean(axis=0) for ids in mapping])
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result, {'unique_chunks': len(unique), 'total_chunks': sum(map(len, mapping)),
                    'max_chunks_per_request': max(map(len, mapping)), 'max_input_wordpieces': max(lengths),
                    'max_encoded_chunk_tokens': max_encoded_length, 'encoder_max_tokens': limit,
                    'truncated_tokens': 0, 'encoding_wall_seconds': time.perf_counter()-started,
                    'wall_time_scope': 'offline CPU batch throughput, not online request latency'}


class TextFeatures:
    def __init__(self, config, model_path=None):
        self.config, self.model_path = config, model_path
        self.encoder = None
        self.pre_columns = []

    def fit_transform(self, frame, cache):
        cache.mkdir(parents=True, exist_ok=True)
        self.pre_columns = ['q_hat_s', 'q_hat_l'] if 'q_hat_s' in frame else []
        fingerprint = digest_bytes('\n'.join(frame.prompt_group).encode('utf-8'))
        if self.config['encoder'] == 'minilm':
            path = Path(self.model_path) if self.model_path else local_minilm_path()
            model_hash = digest(path / 'model.safetensors')
            self.model_path = str(path.resolve())
            key = fingerprint + model_hash
            meta_path, values_path = cache / 'minilm_cache.json', cache / 'minilm_embeddings.npy'
            if meta_path.exists() and values_path.exists() and json.loads(meta_path.read_text(encoding='utf-8')).get('cache_key') == key:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                if digest(values_path) != meta['embedding_file_sha256']:
                    raise ValueError('Embedding cache fingerprint mismatch')
                semantic = np.load(values_path, allow_pickle=False)
                print('Reused frozen MiniLM feature cache', flush=True)
            else:
                self.encoder = load_encoder(path, self.config['torch_threads'])
                semantic, meta = chunk_embeddings(frame.prompt.tolist(), self.encoder, self.config['embedding_batch_size'])
                np.save(values_path, semantic, allow_pickle=False)
                meta.update({'cache_key': key, 'embedding_file_sha256': digest(values_path), 'model_path': str(path),
                             'model_sha256': model_hash, 'dimension': int(semantic.shape[1]),
                             'model_config_sha256': digest(path / 'config.json'),
                             'tokenizer_sha256': digest(path / 'tokenizer.json'),
                             'pooling': 'equal mean of normalized token-chunk vectors, then normalization'})
                save_json(meta_path, meta)
            self.encoder = None
        elif self.config['encoder'] == 'tfidf_svd':
            self.vectorizer = TfidfVectorizer(max_features=self.config['tfidf_max_features'], ngram_range=(1,2), sublinear_tf=True)
            train = frame.split.eq('train').to_numpy()
            sparse_train = self.vectorizer.fit_transform(frame.loc[train, 'prompt'])
            self.svd = TruncatedSVD(n_components=min(self.config['svd_dimensions'], sparse_train.shape[1]-1), random_state=self.config['seed'])
            self.svd.fit(sparse_train)
            semantic = self.svd.transform(self.vectorizer.transform(frame.prompt)).astype(np.float32)
            meta = {'kind': 'train_only_tfidf_svd_not_a_pretrained_transformer', 'dimension': int(semantic.shape[1])}
        else:
            raise ValueError('Unknown explicitly selected encoder')
        raw = self.combine(frame, semantic)
        self.scaler = StandardScaler().fit(raw[frame.split.eq('train')])
        self.semantic_dimension = semantic.shape[1]
        self.feature_names = ['embedding_%03d' % i for i in range(semantic.shape[1])] + STAT_NAMES
        if self.pre_columns:
            self.feature_names += ['q_hat_s', 'q_hat_l', 'pre_predicted_gain']
        self.meta = {**meta, 'input_features': self.feature_names, 'pre_features_present': bool(self.pre_columns),
                     'fitted_scaler_on': 'training_requests_only', 'forbidden_inputs': ['eval_name','request_id','responses','true_scores','true_current_cost','output_length']}
        return self.scaler.transform(raw).astype(np.float32)

    def combine(self, frame, semantic):
        parts = [semantic, statistics(frame.prompt)]
        if self.pre_columns:
            if not set(self.pre_columns) <= set(frame.columns):
                raise ValueError('This model requires first-question Pre features')
            parts += [frame[self.pre_columns].to_numpy(), (frame.q_hat_l-frame.q_hat_s).to_numpy()[:,None]]
        return np.hstack(parts).astype(np.float32)

    def transform(self, frame):
        if self.config['encoder'] == 'minilm':
            if self.encoder is None:
                self.encoder = load_encoder(self.model_path, self.config['torch_threads'])
            semantic, _ = chunk_embeddings(frame.prompt.tolist(), self.encoder, self.config['embedding_batch_size'])
        else:
            semantic = self.svd.transform(self.vectorizer.transform(frame.prompt)).astype(np.float32)
        return self.scaler.transform(self.combine(frame, semantic)).astype(np.float32)


def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()
