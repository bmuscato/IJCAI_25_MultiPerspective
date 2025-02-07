import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from lime.lime_text import LimeTextExplainer
import shap
from captum.attr import LayerConductance

class TopK:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.dataset = load_dataset(dataset_name)
        self.df = pd.DataFrame(self.dataset)

    def average_sentence_length(self, text_column="merged_text"):
        token_lengths = self.df[text_column].apply(lambda x: len(self.tokenizer.tokenize(x)))
        return token_lengths.mean()

    def predict_proba(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            self.model.eval()
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def calculate_dynamic_k(self, text):
        token_length = len(self.tokenizer.tokenize(text))
        return int(token_length / 4)

class LimeExplainer:
    def __init__(self, model, tokenizer, class_names):
        self.explainer = LimeTextExplainer(class_names=class_names)
        self.model = model
        self.tokenizer = tokenizer

    def explain_instance(self, text, predict_proba, k):
        explanation = self.explainer.explain_instance(text, predict_proba, num_features=k)
        return explanation

class ShapExplainer:
    def __init__(self, model, tokenizer):
        self.pred = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0,
            return_all_scores=True,
        )
        self.explainer = shap.Explainer(self.pred)

    def explain(self, text_data):
        shap_values = self.explainer(text_data)
        return shap_values

class LayerConductanceAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prepare_inputs(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        return inputs["input_ids"], inputs["attention_mask"]

    def model_forward(self, inputs_embeds, attention_mask, target_class):
        logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        return logits[:, target_class]

    def layerwise_analysis(self, text, token_to_explain):
        input_ids, attention_mask = self.prepare_inputs(text)
        embeddings = self.model.bert.embeddings(input_ids)
        ref_embeddings = torch.zeros_like(embeddings)

        layer_attrs = []
        for i in range(self.model.config.num_hidden_layers):
            lc = LayerConductance(self.model_forward, self.model.bert.encoder.layer[i])
            layer_attributions = lc.attribute(
                inputs=embeddings,
                baselines=ref_embeddings,
                additional_forward_args=(attention_mask, 0)
            )
            summarized_attr = layer_attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
            layer_attrs.append(summarized_attr)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        plt.figure(figsize=(15, 5))
        sns.heatmap(
            np.array(layer_attrs),
            xticklabels=tokens,
            yticklabels=list(range(1, self.model.config.num_hidden_layers + 1)),
            cmap="coolwarm",
            linewidths=0.2
        )
        plt.title(f"Layer-wise attribution for token '{tokens[token_to_explain]}'")
        plt.xlabel("Tokens")
        plt.ylabel("Layers")
        plt.show()

class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def visualize_token2token_scores(self, scores_mat, all_tokens, x_label_name='Head'):
        num_heads = len(scores_mat)
        num_cols = 3
        num_rows = (num_heads + num_cols - 1) // num_cols

        fig = plt.figure(figsize=(20, num_rows * 5))
        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(num_rows, num_cols, idx + 1)
            im = ax.imshow(scores_np, cmap='viridis')
            fontdict = {'fontsize': 10}
            ax.set_xticks(range(len(all_tokens)))
            ax.set_yticks(range(len(all_tokens)))
            ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(all_tokens, fontdict=fontdict)
            ax.set_xlabel('{} {}'.format(x_label_name, idx + 1))
            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    def visualize_token2head_scores(self, scores_mat, all_tokens):
        fig = plt.figure(figsize=(30, 50))
        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(6, 2, idx + 1)
            im = ax.matshow(scores_np, cmap='viridis')
            fontdict = {'fontsize': 20}
            ax.set_xticks(range(len(all_tokens)))
            ax.set_yticks(range(len(scores)))
            ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(range(len(scores[0])), fontdict=fontdict)
            ax.set_xlabel('Layer {}'.format(idx + 1))
            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

# Example usage
model_name = "../your_model"
dataset_name = "your_dataset"

text_analyzer = TopK(model_name, dataset_name)
avg_sentence_length = text_analyzer.average_sentence_length()
k = int(avg_sentence_length / 4)

lime_explainer = LimeExplainer(text_analyzer.model, text_analyzer.tokenizer, ['Class 0', 'Class 1'])
shap_explainer = ShapExplainer(text_analyzer.model, text_analyzer.tokenizer)
layer_conductance_analyzer = LayerConductanceAnalyzer(text_analyzer.model, text_analyzer.tokenizer)
attention_visualizer = AttentionVisualizer(text_analyzer.model, text_analyzer.tokenizer)

# LIME explanations
for i, row in text_analyzer.df.iterrows():
    text = row['merged_text']
    k = text_analyzer.calculate_dynamic_k(text)
    explanation = lime_explainer.explain_instance(text, text_analyzer.predict_proba, k)
    print(f"Explanation for instance {i} using LIME:")
    explanation.show_in_notebook()
    print(f"Dynamic k value for this instance: {k}\n")

# SHAP explanations
text_data = text_analyzer.df["merged_text"].tolist()
shap_values = shap_explainer.explain(text_data)
shap.plots.text(shap_values)

# Layer Conductance analysis
example_text = text_analyzer.df.iloc[0]["merged_text"]
token_to_explain = 5
layer_conductance_analyzer.layerwise_analysis(example_text, token_to_explain)

# Attention visualization
example_text = text_analyzer.dataset[7]['merged_text']
inputs = text_analyzer.tokenizer(example_text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = text_analyzer.model(**inputs)
    output_attentions_all = outputs.attentions

all_tokens = text_analyzer.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
layer = 3
attention_visualizer.visualize_token2token_scores(output_attentions_all[layer].squeeze().detach().cpu().numpy(), all_tokens)
