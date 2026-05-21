import os.path
from cleantext import clean
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import AutoTokenizer
from imblearn.under_sampling import RandomUnderSampler


def clean_text(text):
    cleaned_text = clean(text,
                         fix_unicode=True,  # fix various unicode errors
                         to_ascii=True,  # transliterate to closest ASCII representation
                         lower=False,  # lowercase text
                         no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
                         no_urls=True,  # replace all URLs with a special token
                         no_emails=True,  # replace all email addresses with a special token
                         no_phone_numbers=True,  # replace all phone numbers with a special token
                         no_numbers=False,  # replace all numbers with a special token
                         no_digits=False,  # replace all digits with a special token
                         no_currency_symbols=False,  # replace all currency symbols with a special token
                         no_punct=False,  # remove punctuations
                         # replace_with_punct="",
                         replace_with_url="<URL>",
                         replace_with_email="<EMAIL>",
                         replace_with_phone_number="<PHONE>",
                         # replace_with_number="<NUMBER>",
                         # replace_with_digit="0",
                         # replace_with_currency_symbol="<CUR>",
                         lang="en"  # set to 'de' for German special handling
                         )
    return cleaned_text

def split_stratify_time(df, test_size, dev_size):
    # split into labels
    df_train = []
    df_test = []
    df_dev = []

    df = df.sort_values(by="Date")

    for label in df['label'].unique():
        df_label = df[df['label'] == label].copy()

        a = int(len(df_label) * (1 - test_size))
        c = int((len(df_label) - a) * dev_size)
        b = len(df_label) - a - c

        split_train_label = df_label.iloc[:a]
        split_dev_label = df_label.iloc[a:a + b]
        split_test_label = df_label.iloc[a + b:a + b + c]

        df_train += [split_train_label]
        df_test += [split_dev_label]
        df_dev += [split_test_label]

    train_split = pd.concat(df_train)
    test_split = pd.concat(df_test)
    dev_split = pd.concat(df_dev)

    return train_split, test_split, dev_split

def reconstruct_page(dataset_df):
    exploded_train = dataset_df[['document_id', 'sentences']].explode('sentences')

    exploded_train['page_idx'] = exploded_train['sentences'].apply(lambda x: x['page_idx'])
    exploded_train['sentence_id'] = exploded_train['sentences'].apply(lambda x: x['sentence_id'])
    exploded_train['block_idx'] = exploded_train['sentences'].apply(lambda x: x['block_idx'])
    exploded_train['text'] = exploded_train['sentences'].apply(lambda x: x['text'])

    page_inputs = exploded_train.groupby(by=['document_id', 'page_idx', 'block_idx'])[
        'text'].sum().reset_index()
    page_inputs = page_inputs.groupby(by=['document_id', 'page_idx'])['text'].apply(lambda x: "\\n".join(x))

    return page_inputs.reset_index()

def get_page_idx(l):
    list_of_lists = [e['page_indices'] for e in l]
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    return list(set(flattened_list))

def get_page_query_map(ds):
    ds_exploded = ds.explode('evidences')
    ds_exploded['page_indices'] = ds_exploded['evidences'].apply(lambda x: x['page_indices'])
    ds_exploded['query'] = ds_exploded['evidences'].apply(lambda x: x['query'])
    mapping = ds_exploded[['document_id', 'page_indices', 'query']].explode('page_indices')
    mapping = mapping.groupby(by=['document_id', 'page_indices'])['query'].apply(lambda x: list(x))
    return mapping.reset_index()

def get_page_stance_map(ds):
    ds_exploded = ds.explode('evidences')
    ds_exploded['page_indices'] = ds_exploded['evidences'].apply(lambda x: x['page_indices'])
    ds_exploded['query'] = ds_exploded['evidences'].apply(lambda x: x['query'])
    ds_exploded['stance'] = ds_exploded['evidences'].apply(lambda x: x['stance'])
    mapping = ds_exploded[['document_id', 'page_indices', 'query', 'stance']].explode('page_indices')
    return mapping.reset_index()


class DatasetBuilder():
    def __init__(self, seed=42):
        print('initializing builder')
        self.seed = seed
        self.train_test_split = 0.2
        self.test_dev_split = 0.5
        self.path = "data"

        self.datasets = {
            # "feverous": self.feverous,
            # "fever": self.fever,
            "averitec": self.averitec,
            "scifact": self.scifact,
            "climatecheck": self.climatecheck,
            "climatefever": self.climatefever,
        }

        self.relation_datasets = {
        #     "climateFEVER_evidence": self.climateFEVER_evidence,
        #     "climateFEVER_evidence_climabench": self.climateFEVER_evidence_climabench
        }

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


    def count_tokens(self, text):
        if not isinstance(text, str) or not text.strip():
            print(f"Invalid text encountered: {text}")
            return 0  # or handle as needed
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)

    def __iter__(self):
        self.dataset_names = iter(self.datasets.keys())
        return self

    def __next__(self):
        try:
            dataset_name = next(self.dataset_names)
        except StopIteration:
            raise StopIteration
        train, test, dev = self.datasets[dataset_name]()

        return (dataset_name, (train, test, dev))

    def train_test_huggingface_datasets(self, raw_dataset, text_column='text', label_column="label"):
        train_dataset = raw_dataset['train']
        
        # Extract full records instead of just text and label
        records = [dict(record) for record in train_dataset]
        labels  = [record[label_column] for record in records]
        
        records_train, records_dev = train_test_split(
            records, test_size=self.train_test_split,
            random_state=self.seed, stratify=labels, shuffle=True
        )

        test_dataset = raw_dataset['test']
        records_test = [dict(record) for record in test_dataset]

        train = pd.DataFrame(records_train)
        test  = pd.DataFrame(records_test)
        dev   = pd.DataFrame(records_dev)

        return train, test, dev


    ############# DATASETS ##################

    def feverous(self):
        raw_dataset = load_dataset('fever/feverous', trust_remote_code=True)
        train = raw_dataset['train'].to_pandas()
        dev = raw_dataset['validation'].to_pandas()
        test = raw_dataset['test'].to_pandas()
        
        train.rename(columns={'claim': 'text'}, inplace=True)
        test.rename(columns={'claim': 'text'}, inplace=True)
        dev.rename(columns={'claim': 'text'}, inplace=True)
        return train, test, dev
    
    def fever(self):
        raw_dataset = load_dataset('fever/fever', 'v1.0', trust_remote_code=True)
        train = raw_dataset['train'].to_pandas()
        dev = raw_dataset['labelled_dev'].to_pandas() # unlabelled_dev, paper_dev
        test = raw_dataset['unlabelled_test'].to_pandas() # paper_test
        
        train.rename(columns={'claim': 'text'}, inplace=True)
        test.rename(columns={'claim': 'text'}, inplace=True)
        dev.rename(columns={'claim': 'text'}, inplace=True)
        return train, test, dev

    def averitec(self):
        # cleaned_datasets/averitec/test.pkl has no labels (blind shared-task set)
        # Use dev.pkl as test, split train.pkl 80/20 for train/val
        folder_path = os.path.join(os.getcwd(), "data", "cleaned_datasets", "averitec")

        train_full = pd.read_parquet(os.path.join(folder_path, "train.pkl"))
        dev        = pd.read_parquet(os.path.join(folder_path, "dev.pkl"))

        for df in (train_full, dev):
            df.drop(columns=[c for c in df.columns if c not in ('text', 'label')], inplace=True)

        from sklearn.model_selection import train_test_split as _tts
        train, test = _tts(train_full, test_size=0.2, random_state=self.seed,
                           stratify=train_full['label'])
        train = train.reset_index(drop=True)
        test  = test.reset_index(drop=True)

        return train, test, dev

    @staticmethod
    def _scifact_claim_label(evidence: dict) -> str:
        """Derive a claim-level veracity label from the per-doc evidence dict.
        Evidence values may be None (no labelled sentences) or an array of
        dicts with 'label' and 'sentences' keys.
        """
        if not evidence:
            return "Not Enough Evidence"
        labels = set()
        for doc in evidence.values():
            if doc is None:
                continue
            for e in doc:
                labels.add(e["label"])
        if "CONTRADICT" in labels:
            return "Refuted"
        if "SUPPORT" in labels:
            return "Supported"
        return "Not Enough Evidence"

    def scifact(self):
        # cleaned_datasets/scifact/test.pkl has no evidence labels (blind set)
        # Use dev.pkl as test, split train.pkl 80/20 for train/val
        folder_path = os.path.join(os.getcwd(), "data", "cleaned_datasets", "scifact")

        train_full = pd.read_parquet(os.path.join(folder_path, "train.pkl"))
        dev        = pd.read_parquet(os.path.join(folder_path, "dev.pkl"))

        for df in (train_full, dev):
            df['label'] = df['evidence'].apply(self._scifact_claim_label)
            df.drop(columns=[c for c in df.columns if c not in ('text', 'label')], inplace=True)

        from sklearn.model_selection import train_test_split as _tts
        train, test = _tts(train_full, test_size=0.2, random_state=self.seed,
                           stratify=train_full['label'])
        train = train.reset_index(drop=True)
        test  = test.reset_index(drop=True)

        return train, test, dev
    
    def climatecheck(self):
        # cleaned_datasets/climatecheck/test.pkl has all empty labels (blind set)
        # Use dev.pkl as test, split train.pkl 80/20 for train/val
        folder_path = os.path.join(os.getcwd(), "data", "cleaned_datasets", "climatecheck")

        train_full = pd.read_parquet(os.path.join(folder_path, "train.pkl"))
        dev        = pd.read_parquet(os.path.join(folder_path, "dev.pkl"))

        for df in (train_full, dev):
            df.drop(columns=[c for c in df.columns if c not in ('text', 'label')], inplace=True)
            df.dropna(subset=['label'], inplace=True)
            df.drop(df[df['label'] == ''].index, inplace=True)

        from sklearn.model_selection import train_test_split as _tts
        train, test = _tts(train_full, test_size=0.2, random_state=self.seed,
                           stratify=train_full['label'])
        train = train.reset_index(drop=True)
        test  = test.reset_index(drop=True)

        return train, test, dev

    def climatefever(self):
        _LABEL_MAP = {
            0: "Supported",
            1: "Refuted",
            2: "Not Enough Evidence",
            3: "Conflicting Evidence/Cherrypicking",
        }
        folder_path = os.path.join(os.getcwd(), "data", "cleaned_datasets", "climatefever")

        train = pd.read_parquet(os.path.join(folder_path, "train.pkl"))
        test  = pd.read_parquet(os.path.join(folder_path, "test.pkl"))
        dev   = pd.read_parquet(os.path.join(folder_path, "dev.pkl"))

        for df in (train, test, dev):
            # One row per claim-evidence pair; deduplicate to claim level by claim_id
            df['label'] = df['claim_label'].map(_LABEL_MAP)
            df.drop_duplicates(subset='claim_id', keep='first', inplace=True)
            df.drop(columns=[c for c in df.columns if c not in ('text', 'label')], inplace=True)

        return train, test, dev

    def climateFEVER_claim(self):
        """
        'DISPUTED', 'NOT_ENOUGH_INFO', 'REFUTES', 'SUPPORTS'
        """

        folder_path = os.path.join(os.getcwd(), "data", "climate-FEVER")

        dataset_path = os.path.join(folder_path, "climate-fever-dataset-r1.jsonl")
        ds = pd.read_json(dataset_path, lines=True)
        texts = ds['claim'].values
        labels = ds['claim_label'].values

        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=self.train_test_split, random_state=self.seed, stratify=labels, shuffle=True)
        X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=self.test_dev_split, random_state=self.seed, stratify=y_test, shuffle=True)

        train = pd.DataFrame({"text": X_train, "label": y_train})
        test = pd.DataFrame({"text": X_test, "label": y_test})
        dev = pd.DataFrame({"text": X_dev, "label": y_dev})

        return train, test, dev

    def climateFEVER_evidence(self):
        folder_path = os.path.join(os.getcwd(), "data", "climate-FEVER")

        dataset_path = os.path.join(folder_path, "climate-fever-dataset-r1.jsonl")
        ds = pd.read_json(dataset_path, lines=True)

        #Todo: use sklearn train_test_split to add stratify
        train_ds = ds.sample(frac=1-self.train_test_split, random_state=self.seed).copy()
        dev_test_ds = ds.drop(train_ds.index).copy()
        test_ds = dev_test_ds.sample(frac=1-self.test_dev_split, random_state=self.seed)
        dev_ds = dev_test_ds.drop(test_ds.index).copy()

        train_ds = train_ds.explode('evidences')
        train_ds['text'] = train_ds['evidences'].apply(lambda x: x['evidence'])
        train_ds['label'] = train_ds['evidences'].apply(lambda x: x['evidence_label'])
        train_ds['hypothesis'] = train_ds['claim']

        test_ds = test_ds.explode('evidences')
        test_ds['text'] = test_ds['evidences'].apply(lambda x: x['evidence'])
        test_ds['label'] = test_ds['evidences'].apply(lambda x: x['evidence_label'])
        test_ds['hypothesis'] = test_ds['claim']

        dev_ds = dev_ds.explode('evidences')
        dev_ds['text'] = dev_ds['evidences'].apply(lambda x: x['evidence'])
        dev_ds['label'] = dev_ds['evidences'].apply(lambda x: x['evidence_label'])
        dev_ds['hypothesis'] = dev_ds['claim']

        train_ds.rename(columns={"text": "text", "hypothesis":"query"}, inplace=True)
        test_ds.rename(columns={"text": "text", "hypothesis":"query"}, inplace=True)
        dev_ds.rename(columns={"text": "text", "hypothesis":"query"}, inplace=True)

        return train_ds, test_ds, dev_ds

    def climateFEVER_evidence_climabench(self):
        folder_path = os.path.join(os.getcwd(), "data", "climate-FEVER")

        dataset_path = os.path.join(folder_path, "climate-fever-dataset-r1.jsonl")
        ds = pd.read_json(dataset_path, lines=True)

        ds = ds.explode('evidences')
        ds['text'] = ds['evidences'].apply(lambda x: x['evidence'])
        ds['label'] = ds['evidences'].apply(lambda x: x['evidence_label'])
        ds['hypothesis'] = ds['claim']

        ds.reset_index(drop=True, inplace=True)

        # Todo: use sklearn train_test_split to add stratify
        train_ds = ds.sample(frac=1 - self.train_test_split, random_state=self.seed).copy()
        dev_test_ds = ds.drop(train_ds.index).copy()
        test_ds = dev_test_ds.sample(frac=1 - self.test_dev_split, random_state=self.seed)
        dev_ds = dev_test_ds.drop(test_ds.index).copy()

        train_ds.rename(columns={"text": "text", "hypothesis": "query"}, inplace=True)
        test_ds.rename(columns={"text": "text", "hypothesis": "query"}, inplace=True)
        dev_ds.rename(columns={"text": "text", "hypothesis": "query"}, inplace=True)

        return train_ds, test_ds, dev_ds


    def filter(self, dataset_df, max_token=4000, min_token=5, drop=True, query=False):

        print(dataset_df.columns)

        dataset_df = dataset_df[dataset_df['token_counts'] < max_token]
        dataset_df = dataset_df[dataset_df['token_counts'] >= min_token]

        if drop == True:
            dataset_df = dataset_df.drop('token_counts', axis=1)
            dataset_df = dataset_df.drop('language', axis=1)
            dataset_df = dataset_df.drop('gibberish', axis=1)
            dataset_df = dataset_df.drop('text', axis=1)
            dataset_df.rename(columns={"clean_text":"text"}, inplace=True)

        return dataset_df

    def prepare_filter(self, dataset_df):
        dataset_df['clean_text'] = dataset_df['text'].apply(clean_text)
        dataset_df['token_counts'] = dataset_df['clean_text'].apply(self.count_tokens)

        return dataset_df

    def weighted_random_sampling(self, data, label_column, n_samples):
        label_distrib = data[label_column].value_counts()
        label_distrib.sort_values(ascending=True, inplace=True)

        sampled_data = []
        n_label = len(label_distrib)

        n = 0
        i = 0
        for label in label_distrib.index:
            label_data = data[data[label_column] == label]

            N_target = int((n_samples - n) / (n_label - i))

            if len(label_data) < N_target:
                sampled_data += [label_data]

                n += len(label_data)
            else:
                sample = label_data.sample(N_target, random_state=self.seed)
                sampled_data += [sample]
                n += len(sample)

            i += 1

        return pd.concat(sampled_data)

    def rebalance(self, df):
        rus = RandomUnderSampler(random_state=self.seed)
        df, _ = rus.fit_resample(X=df, y=df[['label']])

        return df

    def resize(self, df, max_size, stratify_on="label"):
        if max_size >= len(df):
            return df
        else:
            return resample(
                df,
                replace=False,
                n_samples=max_size,
                random_state=self.seed,
                stratify=df[stratify_on]
            )

    def truncate(self, train, dev, max_size=10000, balanced="balanced", stratify_on="label"):
        """
        Truncate the dataset to reduce its size.

        :param train: training dataset (pandas DataFrame)
        :param dev: validation/dev dataset (pandas DataFrame)
        :param max_size: Limit the number of examples in the dataset (int)
        :param max_size_per_label: Limit the number of examples per label (int)
        :param balanced: False to keep the original distribution and True to balance the labels (bool)
        :return: truncated train and dev datasets (tuple of pandas DataFrames)
        """

        if balanced=="random":
            train = self.rebalance(train)
            dev = self.rebalance(dev)

            train = self.resize(train, max_size, stratify_on)
            dev = self.resize(dev, max_size, stratify_on)
        elif balanced=="weighted":
            train = self.weighted_random_sampling(train, 'label', max_size)

            if len(dev) > max_size:
                dev = self.weighted_random_sampling(dev, 'label', max_size)
        else:
            train = self.resize(train, max_size, stratify_on)
            dev = self.resize(dev, max_size, stratify_on)


        return train, dev

    def get_inputs_names(self, dataset_name):
        if dataset_name in self.relation_datasets.keys():
            return ['text', 'query']
        if dataset_name in self.stance_datasets.keys():
            return ['text', 'query']
        else:
            return 'text'
