import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
def compute_recall(image_embeddings, text_embeddings, k_values=[1, 5, 10], batch_size=128):
    """
    Compute recall@k for Flickr30k dataset.

    Args:
        image_embeddings (torch.Tensor): The image embeddings of shape (N, D), where N is the number of images.
        text_embeddings (torch.Tensor): The text embeddings of shape (N, D), where N is the number of texts.
        k_values (list): List of k values for which to compute recall.
        batch_size (int): The batch size for processing embeddings to avoid memory issues.

    Returns:
        dict: A dictionary with recall@k values for image-to-text and text-to-image retrieval.
    """
    assert image_embeddings.shape[0] == text_embeddings.shape[0], "Number of image and text embeddings must be the same"
    num_samples = image_embeddings.shape[0]

    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_embeddings = image_embeddings.to(device)
    text_embeddings = text_embeddings.to(device)

    # Create DataLoader for batching
    image_dataset = TensorDataset(image_embeddings)
    text_dataset = TensorDataset(text_embeddings)
    image_loader = DataLoader(image_dataset, batch_size=batch_size)
    text_loader = DataLoader(text_dataset, batch_size=batch_size)

    # Initialize recall counters
    recall_results = {f'image_to_text_recall@{k}': 0 for k in k_values}
    recall_results.update({f'text_to_image_recall@{k}': 0 for k in k_values})

    # Image-to-Text Recall
    for image_batch in image_loader:
        image_batch = image_batch[0].to(device)  # Extract tensor from tuple and move to device
        similarities = []
        for text_batch in text_loader:
            text_batch = text_batch[0].to(device)  # Extract tensor from tuple and move to device
            similarity = torch.matmul(image_batch, text_batch.T)
            similarities.append(similarity)
        similarity_matrix = torch.cat(similarities, dim=1)

        for i in range(image_batch.size(0)):
            sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
            for k in k_values:
                if i in sorted_indices[:k]:
                    recall_results[f'image_to_text_recall@{k}'] += 1

    # Text-to-Image Recall
    for text_batch in text_loader:
        text_batch = text_batch[0].to(device)  # Extract tensor from tuple and move to device
        similarities = []
        for image_batch in image_loader:
            image_batch = image_batch[0].to(device)  # Extract tensor from tuple and move to device
            similarity = torch.matmul(text_batch, image_batch.T)
            similarities.append(similarity)
        similarity_matrix = torch.cat(similarities, dim=1)

        for i in range(text_batch.size(0)):
            sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
            for k in k_values:
                if i in sorted_indices[:k]:
                    recall_results[f'text_to_image_recall@{k}'] += 1

    # Calculate final recall values
    for k in k_values:
        recall_results[f'image_to_text_recall@{k}'] /= (num_samples*0.01)
        recall_results[f'text_to_image_recall@{k}'] /= (num_samples*0.01)

    return recall_results


def compute_flickr30k_recall(texts_emb, images_emb,  ind_t_img, recall_k_list=[1, 5], batch_size=32, device='cpu'):
    """
    Compute recall@K for Flickr30k dataset for image-to-text and text-to-image retrieval.

    :param texts_emb: Tensor of text embeddings of shape (nb_texts, embedding_dim)
    :param images_emb: Tensor of image embeddings of shape (nb_images, embedding_dim)
    :param recall_k_list: List of recall@k values to compute
    :param batch_size: Batch size for processing
    :param device: Device to use for computation ('cpu' or 'cuda')
    :param data_pairs: List of tuples representing (image_index, text_indices) for positive pairs
    :return: Dictionary containing recall metrics
    """
    # Ensure embeddings are on the correct device
    texts_emb = F.normalize(texts_emb, dim=-1)
    images_emb = F.normalize(images_emb, dim=-1)
    texts_emb = texts_emb.to(device)
    images_emb = images_emb.to(device)
    
    # Calculate similarity scores between text and image embeddings
    scores = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), ind_t_img] = True
    metrics = {}

    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: Compatibility score between text and image embeddings (nb_texts, nb_images)
    :param k: Number of images to consider per text, for retrieval
    :param positive_pairs: Boolean matrix of positive pairs (nb_texts, nb_images)
    :return: Recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # For each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # Compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # One-hot encode the top-k indices, ensure correct dtype
    topk_indices_onehot = F.one_hot(topk_indices, num_classes=nb_images).to(dtype=torch.float32)
    # Reshape positive pairs for broadcasting
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images).to(dtype=torch.float32)
    # Compute number of true positives
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # Compute recall at k
    recall_at_k = nb_true_positive / nb_positive
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    """
    Split the data into batches and apply the function to each batch
    :param func: Function to apply to each batch
    :param X: First input tensor
    :param Y: Second input tensor
    :param batch_size: Batch size
    :param device: Device to use for computation
    :return: Concatenated results from each batch
    """
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


class ImageTextRetrievalTester:
    def __init__(self, model, val_loader, device, img_enc, txt_enc):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.img_enc = img_enc
        self.txt_enc = txt_enc

    def extract_features(self):
        image_features = []
        text_features = []
        
        self.model.eval()
        with torch.no_grad():
            for idx, (images, captions, _, _) in enumerate(self.val_loader):
                images = images.to(self.device)
                output = self.model(self.img_enc(images), self.txt_enc(captions))
                # Assuming your model has methods to extract image and text embeddings
                
                image_features.append(output['image_embedding'].cpu().numpy())
                text_features.append(output['caption_embedding'].cpu().numpy())
                
        image_features = np.vstack(image_features)
        text_features = np.vstack(text_features)
        
        return image_features, text_features

    def compute_retrieval_metrics(self, query_features, database_features):
        similarities = cosine_similarity(query_features, database_features)
        sorted_indices = np.argsort(-similarities, axis=1)  # Sort in descending order by similarity

        # Calculate recall metrics (e.g., R@1, R@5, R@10)
        num_queries = query_features.shape[0]
        recalls = {1: 0, 5: 0, 10: 0}
        
        for idx in range(num_queries):
            ranking = sorted_indices[idx]
            if idx in ranking[:1]:
                recalls[1] += 1
            if idx in ranking[:5]:
                recalls[5] += 1
            if idx in ranking[:10]:
                recalls[10] += 1
        
        for k in recalls:
            recalls[k] = recalls[k]*100 / num_queries
        
        return recalls

    def test_image_to_text(self, image_features, text_features):
        print("Testing Image to Text Retrieval...")
        recalls = self.compute_retrieval_metrics(image_features, text_features)
        print(f"Image-to-Text Recall@1: {recalls[1]:.4f}, Recall@5: {recalls[5]:.4f}, Recall@10: {recalls[10]:.4f}")

    def test_text_to_image(self, image_features, text_features):
        print("Testing Text to Image Retrieval...")
        recalls = self.compute_retrieval_metrics(text_features, image_features)
        print(f"Text-to-Image Recall@1: {recalls[1]:.4f}, Recall@5: {recalls[5]:.4f}, Recall@10: {recalls[10]:.4f}")

    def run_tests(self):
        image_features, text_features = self.extract_features()
        self.test_image_to_text(image_features, text_features)
        self.test_text_to_image(image_features, text_features)

# Usage example:
# tester = ImageTextRetrievalTester(model, val_loader, device)
# tester.run_tests()