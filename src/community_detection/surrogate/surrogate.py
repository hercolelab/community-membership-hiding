import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
import torch.nn as nn
from src.graph_environment.env import GraphEnvironment
from src.utils.utils import Utils
from src.community_detection.surrogate.surrogate_training import SurrogateGNN, from_ig_graph_to_geometric_data
import argparse


def load_model(graph_name, n_nodes, n_clusters, device='cpu'):
    model = SurrogateGNN(n_nodes, n_clusters)
    model_path = f'src/community_detection/surrogate/models/surrogate_gnn_{graph_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_clusters(model, graph, device='cpu'):
    data = from_ig_graph_to_geometric_data(graph)
    data = data.to(device)
    with torch.no_grad():
        logits = model(data)  # [n_nodes, n_clusters]
        pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
    return pred_labels

def main():
    parser = argparse.ArgumentParser(description='Esegui clustering con SurrogateGNN')
    parser.add_argument('--graph_name', type=str, required=True, help='Nome del grafo (es. KAR)')
    parser.add_argument('--alg', type=str, default='LEID', help='Algoritmo di env (default: LEID)')
    parser.add_argument('--graph_path', type=str, default=None, help='Path opzionale al file del grafo')
    args = parser.parse_args()

    # Carica grafo e env
    if args.graph_path:
        graph = Utils.import_graph(args.graph_path)
        n_nodes = graph.vcount()
        env = GraphEnvironment(args.graph_name, [args.alg])
    else:
        env = GraphEnvironment(args.graph_name, [args.alg])
        graph = env.original_graph.copy()
        n_nodes = graph.vcount()

    # Determina n_clusters dal modello salvato (puoi anche passarlo come argomento o dedurlo da un file di config)
    # Qui assumiamo n_clusters = max label + 1 dal training set
    # Per sicurezza, puoi caricare il dataset di training e calcolare n_clusters
    # Qui esempio statico:
    n_clusters = 4  # Sostituisci con il valore corretto per il tuo grafo

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.graph_name, n_nodes, n_clusters, device=device)
    pred_labels = predict_clusters(model, graph, device=device)
    print(f"Cluster labels per ogni nodo: {pred_labels}")

if __name__ == "__main__":
    main() 