
"""
DCScore function
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, chi2_kernel, polynomial_kernel, laplacian_kernel
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from torchvision.models import inception_v3
from tqdm import tqdm
import torchvision.transforms as transforms
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# # Add preprocessing to dataset transforms
class TransformedIndexDataset(torch.utils.data.Dataset):
    """Wrapper class for dataset to add preprocess transform"""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, labels = self.dataset[idx]
        img = self.transform(img)
        return img, labels, idx

    def __len__(self):
        return len(self.dataset)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def to_batches(lst, batch_size):
    batches = []
    i = 0
    while i < len(lst):
        batches.append(lst[i : i + batch_size])
        i += batch_size
    return batches

class DCScore:
    def __init__(self, embedder_path: str='./model_weights/unsup-simcse-bert-base-uncased', 
                 device=None
                ):
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        elif type(device) == str:
            self.device = torch.device(device)

        # load model
        self.embedder_path = embedder_path
        if 'LaBSE' in embedder_path:
            self.model = SentenceTransformer(embedder_path)
        elif 'inception' in embedder_path:
            self.features_size = 2048
            self.model = inception_v3(weights='IMAGENET1K_V1')
            self.model.fc = torch.nn.Identity()
            self.model = self.model.to(self.device)
            self.model.eval()
            self.preprocess = transforms.Compose(
                                                    [
                                                        transforms.Resize(
                                                            (299, 299), interpolation=transforms.InterpolationMode.BICUBIC
                                                        ),
                                                        transforms.CenterCrop(299),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                                                        ),
                                                    ]
                                                )
        elif 'dinov2' in embedder_path:
            self.features_size = 768
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(
                        (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.model.eval()
            self.model.to("cuda")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(embedder_path, use_fast=True, trust_remote_code=True)
            if "glm" in embedder_path.lower():
                model = AutoModel.from_pretrained(embedder_path, device_map='auto', trust_remote_code=True)
                self.model = model.to(torch.float32)
                self.model = self.model.eval().to(device)
                self.decoder_only = True
            elif "llama" in embedder_path.lower() or "gpt" in embedder_path.lower():
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(embedder_path, trust_remote_code=True, device_map='auto').eval()
                self.decoder_only = True
            else:
                self.decoder_only = False
                self.model = AutoModel.from_pretrained(embedder_path, trust_remote_code=True).eval().to(self.device)

    def get_embedding(self, sents_list, batch_size=10):
        if 'LaBSE' in self.embedder_path:
            embeddings_all = self.model.encode(sents_list)
        elif 'inception' in self.embedder_path or 'dinov2' in self.embedder_path:
            size = len(sents_list)
            embeddings_all = torch.zeros(size, self.features_size, device="cpu")
            idxes = torch.zeros(size, device="cpu")
            sents_list = TransformedIndexDataset(sents_list, self.preprocess)
            dataloader = torch.utils.data.DataLoader(
                                                    sents_list,
                                                    batch_size=batch_size,
                                                    drop_last=False,
                                                    num_workers=4,
                                                    shuffle=True,
                                                )
            start_idx = 0
            with torch.no_grad():
                for img_batch, _, idx in tqdm(dataloader, leave=False, total= int(size // batch_size) + 1):
                    input_data = img_batch.to(self.device)
                    feature = self.model(input_data).cpu()
        
                    # If going to overflow, just get required amount and break
                    if size and start_idx + feature.shape[0] > size:
                        embeddings_all[start_idx:] = feature[: size - start_idx]
                        break
        
                    embeddings_all[start_idx : start_idx + feature.shape[0]] = feature
                    idxes[start_idx : start_idx + feature.shape[0]] = idx
        
                    start_idx = start_idx + feature.shape[0]
                    del input_data, feature
                    torch.cuda.empty_cache()
            # print(f"shape of embeddings_all:{embeddings_all.shape}")
            embeddings_all = np.array(embeddings_all)
        else:
            embeddings = []
            for batch in to_batches(sents_list, batch_size):
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    if self.decoder_only:
                        output = self.model(**inputs, output_hidden_states=True)
                    else:
                        output = self.model(**inputs)

                    # when model is all-mpnet-base-v2, getting embedding using mean pooling.
                    if 'mpnet' in self.embedder_path:
                        sentence_embeddings = mean_pooling(output, inputs['attention_mask'])

                        # Normalize embeddings
                        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                        embeddings.append(sentence_embeddings.squeeze().cpu().numpy())
                        continue
                        
                    # completely version
                    if self.decoder_only:
                        if hasattr(output, "last_hidden_state"):
                            output = output.last_hidden_state[:, -1]
                        elif 'glm' in self.embedder_path:
                            output = output.hidden_states[-1]
                            output = output[-1, :]
                        else:
                            output = output.hidden_states[-1]
                            output = output[:, -1, :]
                    elif 'bge' in self.embedder_path:
                        output = output[0][:, 0]
                    else:
                        if hasattr(output, "pooler_output"):
                            output = output.pooler_output
                        else:
                            output = output.last_hidden_state[:, 0]
                            
                if type(output) == list:
                    output = output[0]
                embeddings.append(output.squeeze().cpu().numpy())
            embeddings_all = np.concatenate(embeddings, 0)
        n, d = embeddings_all.shape
        return embeddings_all, n, d
        
    def calculate_dcscore_by_texts(self, texts_list, batch_size=10, tau=1):
        embeddings_all, n, d = self.get_embedding(texts_list, batch_size)
        embeddings_all = preprocessing.normalize(embeddings_all, axis=1)
        sim_product = torch.from_numpy((embeddings_all @ embeddings_all.T) / tau)
        sim_probs = sim_product.softmax(dim=-1)
        diversity = torch.sum(torch.diag(sim_probs))
        return diversity.item()
        
    def calculate_dcscore_by_embedding(self, embeddings_arr, kernel_type='cs', tau=1):
        if kernel_type == 'cs':
            # cosine similarity as teh kernel function
            # embeddings_arr = preprocessing.normalize(embeddings_arr, axis=1)
            sim_product = torch.from_numpy((embeddings_arr @ embeddings_arr.T) / tau)
            sim_probs = sim_product.softmax(dim=-1)
            diversity = torch.sum(torch.diag(sim_probs))
        elif kernel_type == 'rbf':
            sim_mat = rbf_kernel(embeddings_arr, embeddings_arr, tau)
            sim_probs = torch.nn.functional.softmax(torch.from_numpy(sim_mat), dim=-1)
            diversity = torch.sum(torch.diag(sim_probs))
        elif kernel_type == 'lap':
            sim_mat = laplacian_kernel(embeddings_arr, embeddings_arr, tau)
            sim_probs = torch.nn.functional.softmax(torch.from_numpy(sim_mat), dim=-1)
            diversity = torch.sum(torch.diag(sim_probs))
        elif kernel_type == 'poly':
            sim_mat = polynomial_kernel(embeddings_arr, embeddings_arr, tau)
            sim_probs = torch.nn.functional.softmax(torch.from_numpy(sim_mat), dim=-1)
            diversity = torch.sum(torch.diag(sim_probs))
        
        return diversity.item()
