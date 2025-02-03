import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import logging
import os
import csv
import time
from tqdm import tqdm
import community as community_louvain
from networkx.algorithms import community as nx_comm
import matplotlib.ticker as mticker

# Configura il logging
logging.basicConfig(
    filename='analysis_log.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def load_data(developers_file, repositories_file, collaborations_file, pull_requests_file):
    """
    Carica i dati dai file CSV.
    """
    try:
        logging.info("Caricamento dei dati iniziato.")
        print("Caricamento dei dati...")
        developers = pd.read_csv(developers_file)
        repositories = pd.read_csv(repositories_file)
        collaborations = pd.read_csv(collaborations_file)
        pull_requests = pd.read_csv(pull_requests_file)
        logging.info("Dati caricati correttamente.")
        return developers, repositories, collaborations, pull_requests
    except Exception as e:
        logging.error(f"Errore nel caricamento dei dati: {e}")
        raise

def analyze_network(collaborations):
    """
    Analizza la rete di collaborazioni utilizzando NetworkX.
    Calcola le metriche di centralità e clustering.
    """
    logging.info("Inizio creazione del grafo dalle collaborazioni.")
    print("Creazione del grafo dalle collaborazioni...")
    G = nx.from_pandas_edgelist(collaborations, 'source', 'target', edge_attr=True)
    logging.info("Grafo creato correttamente.")
    
    num_nodes = G.number_of_nodes()
    logging.info(f"Numero di nodi nel grafo: {num_nodes}")
    print(f"Numero di nodi nel grafo: {num_nodes}")
    
    if num_nodes < 2:
        logging.warning(f"Numero di nodi ({num_nodes}) troppo piccolo per analizzare la rete.")
        print(f"Numero di nodi ({num_nodes}) troppo piccolo per analizzare la rete.")
        return G, pd.DataFrame()
    
    # Decidi il valore di k in base al numero di nodi
    k = min(1000, num_nodes - 1)
    if num_nodes <= 1000:
        logging.info(f"Calcolo esatto della betweenness centrality.")
        print(f"Calcolo esatto della betweenness centrality...")
        betweenness_centrality = nx.betweenness_centrality(G, seed=42)
    else:
        logging.info(f"Calcolo approssimato della betweenness centrality con k={k}.")
        print(f"Calcolo approssimato della betweenness centrality con k={k}...")
        betweenness_centrality = nx.betweenness_centrality(G, k=k, seed=42)
    
    logging.info("Calcolo della degree centrality.")
    print("Calcolo della degree centrality...")
    degree_centrality = nx.degree_centrality(G)
    
    logging.info("Calcolo del closeness centrality.")
    print("Calcolo del closeness centrality...")
    closeness_centrality = nx.closeness_centrality(G)
    
    logging.info("Calcolo dell'eigenvector centrality.")
    print("Calcolo dell'eigenvector centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.NetworkXError as e:
        logging.error(f"Errore nel calcolo dell'eigenvector centrality: {e}")
        eigenvector_centrality = {node: 0 for node in G.nodes()}
    
    logging.info("Calcolo del clustering coefficient.")
    print("Calcolo del clustering coefficient...")
    clustering_coefficient = nx.clustering(G)
    
    logging.info("Calcolo del PageRank.")
    print("Calcolo del PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85)
    
    logging.info("Calcolo dell'assortativity.")
    print("Calcolo dell'assortativity...")
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except ZeroDivisionError:
        assortativity = 0  # Assumiamo assortatività zero se non definita
    logging.info(f"Assortativity coefficient: {assortativity}")
    print(f"Assortativity coefficient: {assortativity}")
    
    # Rilevazione delle comunità con diversi algoritmi
    logging.info("Inizio rilevazione delle comunità con l'algoritmo di Louvain.")
    print("Rilevazione delle comunità con l'algoritmo di Louvain...")
    partition_louvain = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition_louvain, 'community_louvain')
    num_communities_louvain = len(set(partition_louvain.values()))
    logging.info(f"Numero di comunità rilevate con Louvain: {num_communities_louvain}")
    print(f"Numero di comunità rilevate con Louvain: {num_communities_louvain}")
    
    logging.info("Inizio rilevazione delle comunità con l'algoritmo Girvan-Newman.")
    print("Rilevazione delle comunità con l'algoritmo Girvan-Newman...")
    try:
        comp_girvan_newman = next(nx_comm.girvan_newman(G))
        partition_girvan_newman = {node: cid for cid, community in enumerate(comp_girvan_newman) for node in community}
        nx.set_node_attributes(G, partition_girvan_newman, 'community_girvan_newman')
        num_communities_girvan_newman = len(set(partition_girvan_newman.values()))
        logging.info(f"Numero di comunità rilevate con Girvan-Newman: {num_communities_girvan_newman}")
        print(f"Numero di comunità rilevate con Girvan-Newman: {num_communities_girvan_newman}")
    except Exception as e:
        logging.error(f"Errore nel calcolo delle comunità con Girvan-Newman: {e}")
        partition_girvan_newman = {}
        num_communities_girvan_newman = 0
        print(f"Errore nel calcolo delle comunità con Girvan-Newman: {e}")
    
    logging.info("Inizio rilevazione delle comunità con l'algoritmo Label Propagation.")
    print("Rilevazione delle comunità con l'algoritmo Label Propagation...")
    try:
        communities_label_propagation = list(nx_comm.label_propagation_communities(G))
        partition_label_propagation = {node: cid for cid, community in enumerate(communities_label_propagation) for node in community}
        nx.set_node_attributes(G, partition_label_propagation, 'community_label_propagation')
        num_communities_label_propagation = len(set(partition_label_propagation.values()))
        logging.info(f"Numero di comunità rilevate con Label Propagation: {num_communities_label_propagation}")
        print(f"Numero di comunità rilevate con Label Propagation: {num_communities_label_propagation}")
    except Exception as e:
        logging.error(f"Errore nel calcolo delle comunità con Label Propagation: {e}")
        partition_label_propagation = {}
        num_communities_label_propagation = 0
        print(f"Errore nel calcolo delle comunità con Label Propagation: {e}")
    
    logging.info("Calcolo delle metriche aggiuntive completato.")
    print("Calcolo delle metriche aggiuntive completato.")
    
    # Aggiungi le metriche al grafo
    nx.set_node_attributes(G, degree_centrality, 'degree_centrality')
    nx.set_node_attributes(G, betweenness_centrality, 'betweenness_centrality')
    nx.set_node_attributes(G, closeness_centrality, 'closeness_centrality')
    nx.set_node_attributes(G, eigenvector_centrality, 'eigenvector_centrality')
    nx.set_node_attributes(G, clustering_coefficient, 'clustering_coefficient')
    nx.set_node_attributes(G, pagerank, 'pagerank')
    nx.set_node_attributes(G, assortativity, 'assortativity_coefficient')
    nx.set_node_attributes(G, partition_louvain, 'community_louvain')
    nx.set_node_attributes(G, partition_girvan_newman, 'community_girvan_newman')
    nx.set_node_attributes(G, partition_label_propagation, 'community_label_propagation')
    
    # Crea un DataFrame con le metriche
    logging.info("Creazione del DataFrame delle metriche.")
    print("Creazione del DataFrame delle metriche...")
    metrics_df = pd.DataFrame({
        'developer': list(G.nodes()),
        'degree_centrality': [degree_centrality[node] for node in G.nodes()],
        'betweenness_centrality': [betweenness_centrality.get(node, 0) for node in G.nodes()],
        'closeness_centrality': [closeness_centrality[node] for node in G.nodes()],
        'eigenvector_centrality': [eigenvector_centrality.get(node, 0) for node in G.nodes()],
        'clustering_coefficient': [clustering_coefficient[node] for node in G.nodes()],
        'pagerank': [pagerank[node] for node in G.nodes()],
        'assortativity_coefficient': [assortativity for _ in G.nodes()],
        'community_louvain': [partition_louvain.get(node, -1) for node in G.nodes()],
        'community_girvan_newman': [partition_girvan_newman.get(node, -1) for node in G.nodes()],
        'community_label_propagation': [partition_label_propagation.get(node, -1) for node in G.nodes()]
    })
    
    # Salva le metriche
    logging.info("Salvataggio delle metriche in 'developer_metrics_analysis.csv'.")
    print("Salvataggio delle metriche in 'developer_metrics_analysis.csv'...")
    metrics_df.to_csv('developer_metrics_analysis.csv', index=False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')
    logging.info("Metriche salvate in 'developer_metrics_analysis.csv'.")
    
    # Esporta le statistiche ordinate in senso crescente
    logging.info("Creazione e salvataggio delle metriche ordinate in senso crescente.")
    print("Creazione e salvataggio delle metriche ordinate in senso crescente...")
    sorted_metrics_df = metrics_df.sort_values(by=['degree_centrality'], ascending=True)
    sorted_metrics_df.to_csv('developer_metrics_analysis_sorted.csv', index=False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')
    logging.info("Metriche ordinate salvate in 'developer_metrics_analysis_sorted.csv'.")
    print("Metriche ordinate salvate in 'developer_metrics_analysis_sorted.csv'.")
    
    return G, metrics_df

def visualize_network(G, metrics_df):
    """
    Visualizza la rete di collaborazioni con NetworkX.
    """
    if metrics_df.empty:
        logging.warning("DataFrame delle metriche vuoto. Salto la visualizzazione della rete.")
        print("DataFrame delle metriche vuoto. Salto la visualizzazione della rete.")
        return
    
    logging.info("Inizio visualizzazione della rete.")
    print("Visualizzazione della rete...")
    plt.figure(figsize=(15, 10))
    
    # Normalizza le metriche per la dimensione dei nodi
    sizes = metrics_df['degree_centrality'] * 3000
    node_colors = metrics_df['pagerank']
    
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.8)
    edges = nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    
    plt.colorbar(nodes, label='PageRank')
    plt.title('Rete delle Collaborazioni tra Sviluppatori')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('collaborations_network.png')
    plt.show()
    logging.info("Visualizzazione della rete salvata in 'collaborations_network.png'.")
    print("Visualizzazione della rete salvata in 'collaborations_network.png'.")

def analyze_repositories(repositories):
    """
    Analizza i repository per ottenere statistiche come distribuzione di stelle, fork, issue, ecc.
    """
    logging.info("Inizio analisi dei repository.")
    print("Analisi dei repository...")
    stats = {
        'stargazers_count': repositories['stargazers_count'].describe(),
        'forks_count': repositories['forks_count'].describe(),
        'open_issues_count': repositories['open_issues_count'].describe(),
        'watchers_count': repositories['watchers_count'].describe(),
        'size': repositories['size'].describe()
    }
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('repository_statistics.csv', sep=',', quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')
    logging.info("Statistiche dei repository salvate in 'repository_statistics.csv'.")
    print("Statistiche dei repository salvate in 'repository_statistics.csv'.")
    
    # Visualizza alcune distribuzioni
    logging.info("Creazione della distribuzione delle stelle.")
    print("Creazione della distribuzione delle stelle...")
    plt.figure(figsize=(10, 6))
    repositories['stargazers_count'].hist(bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribuzione delle Stelle per Repository')
    plt.xlabel('Stelle')
    plt.ylabel('Numero di Repository')
    plt.tight_layout()
    plt.savefig('stars_distribution.png')
    plt.show()
    logging.info("Distribuzione delle stelle salvata in 'stars_distribution.png'.")
    print("Distribuzione delle stelle salvata in 'stars_distribution.png'.")

def analyze_pull_requests(pull_requests):
    """
    Analizza le Pull Requests per ottenere statistiche come numero di aggiunte, cancellazioni, ecc.
    """
    logging.info("Inizio analisi delle Pull Requests.")
    print("Analisi delle Pull Requests...")
    stats = {
        'additions': pull_requests['additions'].describe(),
        'deletions': pull_requests['deletions'].describe(),
        'changed_files': pull_requests['changed_files'].describe(),
        'comments_count': pull_requests['comments_count'].describe(),
        'reviews_count': pull_requests['reviews_count'].describe()
    }
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('pull_requests_statistics.csv', sep=',', quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')
    logging.info("Statistiche delle Pull Requests salvate in 'pull_requests_statistics.csv'.")
    print("Statistiche delle Pull Requests salvate in 'pull_requests_statistics.csv'.")
    
    # Visualizza alcune distribuzioni
    logging.info("Creazione della distribuzione delle aggiunte nelle Pull Requests.")
    print("Creazione della distribuzione delle aggiunte nelle Pull Requests...")
    plt.figure(figsize=(10, 6))
    pull_requests['additions'].hist(bins=50, color='salmon', edgecolor='black')
    plt.title('Distribuzione delle Aggiunte nelle Pull Requests')
    plt.xlabel('Aggiunte')
    plt.ylabel('Numero di Pull Requests')
    plt.tight_layout()
    plt.savefig('additions_distribution.png')
    plt.show()
    logging.info("Distribuzione delle aggiunte salvata in 'additions_distribution.png'.")
    print("Distribuzione delle aggiunte salvata in 'additions_distribution.png'.")
    
    # Nuovo grafico: Distribuzione delle deletions
    logging.info("Creazione della distribuzione delle cancellazioni nelle Pull Requests.")
    print("Creazione della distribuzione delle cancellazioni nelle Pull Requests...")
    plt.figure(figsize=(10, 6))
    pull_requests['deletions'].hist(bins=50, color='lightgreen', edgecolor='black')
    plt.title('Distribuzione delle Cancellazioni nelle Pull Requests')
    plt.xlabel('Cancellazioni')
    plt.ylabel('Numero di Pull Requests')
    plt.tight_layout()
    plt.savefig('deletions_distribution.png')
    plt.show()
    logging.info("Distribuzione delle cancellazioni salvata in 'deletions_distribution.png'.")
    print("Distribuzione delle cancellazioni salvata in 'deletions_distribution.png'.")

def analyze_additional_metrics(G, metrics_df):
    """
    Analizza ulteriori metriche della rete e crea nuovi grafici.
    """
    logging.info("Inizio analisi delle metriche aggiuntive della rete.")
    print("Analisi delle metriche aggiuntive della rete...")
    
    # Distribuzione della betweenness centrality
    logging.info("Creazione della distribuzione della betweenness centrality.")
    print("Creazione della distribuzione della betweenness centrality...")
    plt.figure(figsize=(10, 6))
    metrics_df['betweenness_centrality'].hist(bins=50, color='purple', edgecolor='black')
    plt.title('Distribuzione della Betweenness Centrality')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Numero di Sviluppatori')
    plt.tight_layout()
    plt.savefig('betweenness_centrality_distribution.png')
    plt.show()
    logging.info("Distribuzione della betweenness centrality salvata in 'betweenness_centrality_distribution.png'.")
    print("Distribuzione della betweenness centrality salvata in 'betweenness_centrality_distribution.png'.")
    
    # Distribuzione del clustering coefficient
    logging.info("Creazione della distribuzione del clustering coefficient.")
    print("Creazione della distribuzione del clustering coefficient...")
    plt.figure(figsize=(10, 6))
    metrics_df['clustering_coefficient'].hist(bins=50, color='orange', edgecolor='black')
    plt.title('Distribuzione del Clustering Coefficient')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Numero di Sviluppatori')
    plt.tight_layout()
    plt.savefig('clustering_coefficient_distribution.png')
    plt.show()
    logging.info("Distribuzione del clustering coefficient salvata in 'clustering_coefficient_distribution.png'.")
    print("Distribuzione del clustering coefficient salvata in 'clustering_coefficient_distribution.png'.")
    
    # Distribuzione del PageRank
    logging.info("Creazione della distribuzione del PageRank.")
    print("Creazione della distribuzione del PageRank...")
    plt.figure(figsize=(10, 6))
    metrics_df['pagerank'].hist(bins=50, color='teal', edgecolor='black')
    plt.title('Distribuzione del PageRank')
    plt.xlabel('PageRank')
    plt.ylabel('Numero di Sviluppatori')
    plt.tight_layout()
    plt.savefig('pagerank_distribution.png')
    plt.show()
    logging.info("Distribuzione del PageRank salvata in 'pagerank_distribution.png'.")
    print("Distribuzione del PageRank salvata in 'pagerank_distribution.png'.")

def visualize_additional_graphs(metrics_df):
    """
    Crea grafici aggiuntivi basati sulle metriche calcolate.
    """
    logging.info("Inizio visualizzazione dei grafici aggiuntivi.")
    print("Visualizzazione dei grafici aggiuntivi...")
    
    # Scatter plot: Betweenness Centrality vs Degree Centrality
    logging.info("Creazione dello scatter plot Betweenness vs Degree Centrality.")
    print("Creazione dello scatter plot Betweenness vs Degree Centrality...")
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['degree_centrality'], metrics_df['betweenness_centrality'], alpha=0.5, color='blue')
    plt.title('Betweenness Centrality vs Degree Centrality')
    plt.xlabel('Degree Centrality')
    plt.ylabel('Betweenness Centrality')
    plt.tight_layout()
    plt.savefig('betweenness_vs_degree_scatter.png')
    plt.show()
    logging.info("Scatter plot Betweenness vs Degree salvato in 'betweenness_vs_degree_scatter.png'.")
    print("Scatter plot Betweenness vs Degree salvato in 'betweenness_vs_degree_scatter.png'.")
    
    # Scatter plot: PageRank vs Degree Centrality
    logging.info("Creazione dello scatter plot PageRank vs Degree Centrality.")
    print("Creazione dello scatter plot PageRank vs Degree Centrality...")
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['degree_centrality'], metrics_df['pagerank'], alpha=0.5, color='red')
    plt.title('PageRank vs Degree Centrality')
    plt.xlabel('Degree Centrality')
    plt.ylabel('PageRank')
    plt.tight_layout()
    plt.savefig('pagerank_vs_degree_scatter.png')
    plt.show()
    logging.info("Scatter plot PageRank vs Degree salvato in 'pagerank_vs_degree_scatter.png'.")
    print("Scatter plot PageRank vs Degree salvato in 'pagerank_vs_degree_scatter.png'.")
    
    # Grafico di confronto tra i risultati delle diverse comunità
    logging.info("Creazione del grafico di confronto tra i risultati delle diverse comunità.")
    print("Creazione del grafico di confronto tra i risultati delle diverse comunità...")
    comparison_data = {
        'Algorithm': ['Louvain', 'Girvan-Newman', 'Label Propagation'],
        'Number of Communities': [
            metrics_df['community_louvain'].nunique(),
            metrics_df['community_girvan_newman'].nunique() if metrics_df['community_girvan_newman'].nunique() > 0 else 0,
            metrics_df['community_label_propagation'].nunique()
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('community_comparation.csv', index=False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')
    
    # Creazione del grafico a barre comparativo
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df['Algorithm'], comparison_df['Number of Communities'], color=['green', 'blue', 'orange'], alpha=0.7)
    plt.title('Confronto del Numero di Comunità Rilevate da Diversi Algoritmi')
    plt.xlabel('Algoritmo di Rilevamento delle Comunità')
    plt.ylabel('Numero di Comunità')
    plt.tight_layout()
    plt.savefig('community_comparison_bar_chart.png')
    plt.show()
    logging.info("Grafico di confronto delle comunità salvato in 'community_comparison_bar_chart.png'.")
    print("Grafico di confronto delle comunità salvato in 'community_comparison_bar_chart.png'.")
    
    # Miglioramento del file 'community_comparation.csv' con descrizioni
    logging.info("Miglioramento del file 'community_comparation.csv' con etichette esplicative.")
    print("Miglioramento del file 'community_comparation.csv' con etichette esplicative...")
    description = {
        'Louvain': 'Algoritmo di Louvain per massimizzare la modularità.',
        'Girvan-Newman': 'Algoritmo Girvan-Newman basato sulla rimozione degli archi di massima betweenness.',
        'Label Propagation': 'Algoritmo di propagazione delle etichette per la rilevazione delle comunità.'
    }
    comparison_df['Description'] = comparison_df['Algorithm'].map(description)
    comparison_df.to_csv('community_comparation_detailed.csv', index=False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')
    logging.info("File 'community_comparation_detailed.csv' creato con descrizioni esplicative.")
    print("File 'community_comparation_detailed.csv' creato con descrizioni esplicative.")

def visualize_community_graph(G, metrics_df):
    """
    Visualizza la rete di collaborazioni con le comunità.
    """
    logging.info("Inizio visualizzazione della rete con le comunità.")
    print("Visualizzazione della rete con le comunità...")
    
    # Layout di base
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)

    # Assegna un colore a ogni comunità di Louvain
    communities_louvain = metrics_df.set_index('developer')['community_louvain'].to_dict()
    unique_communities_louvain = list(set(communities_louvain.values()))
    colors = plt.cm.tab20.colors  # Usa una colormap con diversi colori
    community_color_map_louvain = {
        community: colors[i % len(colors)] for i, community in enumerate(unique_communities_louvain)
    }
    node_colors_louvain = [community_color_map_louvain[communities_louvain[node]] for node in G.nodes()]
    
    # Visualizzazione per Louvain
    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(
        G, pos,
        node_size=metrics_df['degree_centrality'] * 3000,
        node_color=node_colors_louvain,
        alpha=0.8  # Rimosso 'cmap=plt.cm.tab20'
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    plt.title('Rete delle Collaborazioni con le Comunità (Louvain)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('collaborations_network_communities_louvain.png')
    plt.show()
    logging.info("Visualizzazione della rete con le comunità (Louvain) salvata in 'collaborations_network_communities_louvain.png'.")
    print("Visualizzazione della rete con le comunità (Louvain) salvata in 'collaborations_network_communities_louvain.png'.")
    
    # Visualizzazione per Girvan-Newman
    communities_girvan_newman = metrics_df['community_girvan_newman'].dropna().unique()
    if len(communities_girvan_newman) > 0:
        community_color_map_girvan_newman = {
            community: colors[i % len(colors)] for i, community in enumerate(communities_girvan_newman)
        }
        node_colors_girvan_newman = [
            community_color_map_girvan_newman[metrics_df.loc[i, 'community_girvan_newman']]
            if not pd.isna(metrics_df.loc[i, 'community_girvan_newman']) else 'black'
            for i in metrics_df.index
        ]
        
        plt.figure(figsize=(15, 10))
        nx.draw_networkx_nodes(
            G, pos,
            node_size=metrics_df['degree_centrality'] * 3000,
            node_color=node_colors_girvan_newman,
            alpha=0.8  # Rimosso 'cmap=plt.cm.tab20'
        )
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
        plt.title('Rete delle Collaborazioni con le Comunità (Girvan-Newman)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('collaborations_network_communities_girvan_newman.png')
        plt.show()
        logging.info("Visualizzazione della rete con le comunità (Girvan-Newman) salvata in 'collaborations_network_communities_girvan_newman.png'.")
        print("Visualizzazione della rete con le comunità (Girvan-Newman) salvata in 'collaborations_network_communities_girvan_newman.png'.")
    else:
        logging.warning("Nessuna comunità rilevata con Girvan-Newman. Salto la visualizzazione.")
        print("Nessuna comunità rilevata con Girvan-Newman. Salto la visualizzazione.")
    
    # Visualizzazione per Label Propagation
    communities_label_propagation = metrics_df['community_label_propagation'].dropna().unique()
    community_color_map_label_propagation = {
        community: colors[i % len(colors)] for i, community in enumerate(communities_label_propagation)
    }
    node_colors_label_propagation = [
        community_color_map_label_propagation[metrics_df.loc[i, 'community_label_propagation']]
        if not pd.isna(metrics_df.loc[i, 'community_label_propagation']) else 'black'
        for i in metrics_df.index
    ]
    
    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(
        G, pos,
        node_size=metrics_df['degree_centrality'] * 3000,
        node_color=node_colors_label_propagation,
        alpha=0.8  # Rimosso 'cmap=plt.cm.tab20'
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    plt.title('Rete delle Collaborazioni con le Comunità (Label Propagation)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('collaborations_network_communities_label_propagation.png')
    plt.show()
    logging.info("Visualizzazione della rete con le comunità (Label Propagation) salvata in 'collaborations_network_communities_label_propagation.png'.")
    print("Visualizzazione della rete con le comunità (Label Propagation) salvata in 'collaborations_network_communities_label_propagation.png'.")

def calculate_network_statistics(G):
    """
    Calcola statistiche di rete aggiuntive.
    """
    logging.info("Calcolo delle statistiche di rete aggiuntive.")
    print("Calcolo delle statistiche di rete aggiuntive...")
    
    try:
        # Transitività
        transitivity = nx.transitivity(G)
        logging.info(f"Transitività: {transitivity}")
        
        # Densità
        density = nx.density(G)
        logging.info(f"Densità della rete: {density}")
        
        # Connettività
        if nx.is_connected(G):
            connectivity = nx.node_connectivity(G)
            logging.info(f"Connettività della rete (node connectivity): {connectivity}")
        else:
            connectivity = "Rete Disconnessa"
            logging.info("La rete è disconnessa.")
        
        # Compattezza (basata sulla transitività)
        compactness = transitivity
        
        # Diametro
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            logging.info(f"Diametro della rete: {diameter}")
        else:
            diameter = "Rete Disconnessa"
            logging.info("La rete è disconnessa. Il diametro non può essere calcolato.")
        
        # Coefficiente di Clustering medio
        clustering = nx.average_clustering(G)
        logging.info(f"Coefficiente di Clustering medio: {clustering}")
        
        # Creazione del dizionario delle statistiche
        stats = {
            'Transitività': [transitivity],
            'Densità della Rete': [density],
            'Connettività (Node Connectivity)': [connectivity],
            'Compattezza (Transitività)': [compactness],
            'Diametro della Rete': [diameter],
            'Coefficiente di Clustering Medio': [clustering]
        }
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv('network_statistics.csv', index=False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')
        logging.info("Statistiche di rete salvate in 'network_statistics.csv'.")
        print("Statistiche di rete salvate in 'network_statistics.csv'.")
        
        # Esporta le statistiche ordinate in senso crescente (per alcune colonne numericabili)
        logging.info("Creazione e salvataggio delle statistiche di rete ordinate in senso crescente.")
        print("Creazione e salvataggio delle statistiche di rete ordinate in senso crescente...")
        # Qui facciamo un tentativo di sort su alcune colonne numericabili (dove i valori non sono stringhe)
        numeric_cols = ['Transitività','Densità della Rete','Compattezza (Transitività)','Coefficiente di Clustering Medio']
        numeric_stats = stats_df[numeric_cols].astype(float)
        sorted_stats_df = stats_df.reindex(numeric_stats.sort_values(by=numeric_cols, ascending=True).index)
        sorted_stats_df.to_csv('network_statistics_sorted.csv', index=False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')
        logging.info("Statistiche di rete ordinate salvate in 'network_statistics_sorted.csv'.")
        print("Statistiche di rete ordinate salvate in 'network_statistics_sorted.csv'.")
        
    except Exception as e:
        logging.error(f"Errore nel calcolo delle statistiche di rete: {e}")
        print(f"Errore nel calcolo delle statistiche di rete: {e}")

def dynamic_assortativity_over_time(collaborations):
    """
    Analizza l'evoluzione dell'assortatività (e altre metriche, se necessario) nell'arco dei due anni.
    Viene creato un grafico lineare che mostra come l'assortatività varia nel tempo.
    """
    logging.info("Inizio analisi dinamica dell'assortatività nel tempo.")
    print("Analisi dinamica dell'assortatività nel tempo...")
    
    # Assicurarsi che esista la colonna 'timestamp'
    if 'timestamp' not in collaborations.columns:
        logging.warning("Nessun 'timestamp' nelle collaborazioni, impossibile calcolare l'assortatività dinamica.")
        print("Nessun 'timestamp' nelle collaborazioni, impossibile calcolare l'assortatività dinamica.")
        return
    
    # Converti i timestamp in formato datetime, rimuovendo il fuso orario se presente
    collaborations['timestamp'] = pd.to_datetime(collaborations['timestamp'], errors='coerce').dt.tz_convert(None)
    # Rimuovi eventuali righe con timestamp NaN
    collaborations = collaborations.dropna(subset=['timestamp'])
    
    if collaborations.empty:
        logging.warning("Nessun dato valido (timestamp mancanti) per l'analisi temporale.")
        print("Nessun dato valido per l'analisi temporale.")
        return
    
    # Raggruppa per mese per vedere l'evoluzione nel tempo
    collaborations['year_month'] = collaborations['timestamp'].dt.to_period('M')
    time_slices = sorted(collaborations['year_month'].unique())
    
    assortativity_over_time = []
    
    for period in time_slices:
        # Filtro le collaborazioni di quel periodo (mese/anno)
        subset = collaborations[collaborations['year_month'] == period]
        # Crea un grafo con le sole collaborazioni di quel periodo
        G_sub = nx.from_pandas_edgelist(subset, 'source', 'target', edge_attr=True)
        
        if G_sub.number_of_nodes() > 1:
            degrees_sub = dict(G_sub.degree())
            unique_degrees = set(degrees_sub.values())
            
            if len(unique_degrees) > 1:
                try:
                    assort_sub = nx.degree_assortativity_coefficient(G_sub)
                except ZeroDivisionError:
                    assort_sub = None  # Assumiamo assortatività non definita
            else:
                # Se tutti i nodi hanno lo stesso grado o c'è un solo nodo
                assort_sub = None
        else:
            assort_sub = None
        
        assortativity_over_time.append((str(period), assort_sub))
    
    # Crea un DataFrame per il grafico, rimuovendo i periodi con assortatività non definita
    assort_df = pd.DataFrame(assortativity_over_time, columns=['period', 'assortativity'])
    assort_df = assort_df.dropna(subset=['assortativity'])
    assort_df.to_csv('dynamic_assortativity_over_time.csv', index=False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL)
    
    # Crea il grafico
    plt.figure(figsize=(12, 6))
    plt.plot(assort_df['period'], assort_df['assortativity'], marker='o', color='blue', linestyle='-')
    
    # Ottimizzazione dell'asse x per rendere le etichette meno affollate
    step = 3  # Mostra un tick ogni 3 periodi (es. 3 mesi)
    if len(assort_df['period']) > 0:
        ticks_to_show = range(0, len(assort_df['period']), step)
        plt.xticks(
            ticks=ticks_to_show, 
            labels=assort_df['period'][::step], 
            rotation=45
        )
    
    plt.title('Evoluzione dell\'Assortatività nel Tempo')
    plt.xlabel('Periodo (Year-Month)')
    plt.ylabel('Assortativity Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dynamic_assortativity_over_time.png')
    plt.show()
    
    logging.info("Analisi dell'assortatività dinamica completata e salvata in 'dynamic_assortativity_over_time.png'.")
    print("Analisi dell'assortatività dinamica completata.")

def developer_metrics_analysis_with_subcommunities(G, metrics_df):
    """
    Analizza le sotto-comunità della rete (componenti connesse o sub-grafi).
    Restituisce e salva le metriche aggregate per ciascuna sotto-comunità.
    Viene inoltre prodotto un grafico a barre con il numero di nodi in ogni sub-componente.
    """
    logging.info("Inizio analisi delle metriche con sotto-comunità (componenti connesse).")
    print("Analisi delle metriche con sotto-comunità...")
    
    # Identifica le componenti connesse (se la rete è non orientata, usa connected_components)
    if nx.is_empty(G):
        logging.warning("Il grafo è vuoto. Nessuna analisi delle sotto-comunità possibile.")
        print("Il grafo è vuoto. Nessuna analisi delle sotto-comunità possibile.")
        return
    
    # Poiché G è non orientato (di default in from_pandas_edgelist), usiamo connected_components
    components = list(nx.connected_components(G))
    
    sub_communities_info = []
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)
        sub_nodes = len(subgraph.nodes())
        sub_edges = len(subgraph.edges())
        
        # Calcolo di metriche base per la sotto-comunità
        density_sub = nx.density(subgraph)
        if sub_nodes > 1:
            degrees_sub = dict(subgraph.degree())
            unique_degrees = set(degrees_sub.values())
            
            if len(unique_degrees) > 1:
                try:
                    assort_sub = nx.degree_assortativity_coefficient(subgraph)
                except ZeroDivisionError:
                    assort_sub = None
            else:
                assort_sub = None
        else:
            assort_sub = None
        
        sub_communities_info.append({
            'subcommunity_id': i,
            'num_nodes': sub_nodes,
            'num_edges': sub_edges,
            'density': density_sub,
            'assortativity': assort_sub
        })
    
    sub_communities_df = pd.DataFrame(sub_communities_info)
    sub_communities_df.to_csv('subcommunities_analysis.csv', index=False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL)
    
    # Grafico a barre con dimensioni delle sub-comunità
    plt.figure(figsize=(12, 6))
    plt.bar(sub_communities_df['subcommunity_id'], sub_communities_df['num_nodes'], color='coral', alpha=0.7)
    plt.title('Dimensione di ogni sotto-comunità (in termini di nodi)')
    plt.xlabel('Sotto-comunità ID')
    plt.ylabel('Numero di nodi')
    plt.tight_layout()
    plt.savefig('subcommunities_size_bar_chart.png')
    plt.show()
    
    logging.info("Analisi delle sotto-comunità completata e salvata in 'subcommunities_analysis.csv'.")
    print("Analisi delle sotto-comunità completata e salvata in 'subcommunities_analysis.csv'.")

def main():
    # Definisci i nomi dei file
    developers_file = 'developers_large.csv'
    repositories_file = 'repositories_large.csv'
    collaborations_file = 'collaborations_large.csv'
    pull_requests_file = 'pull_requests_large.csv'

    # Carica i dati
    logging.info("Inizio caricamento dei dati.")
    print("Caricamento dei dati...")
    try:
        developers, repositories, collaborations, pull_requests = load_data(
            developers_file, repositories_file, collaborations_file, pull_requests_file
        )
    except Exception as e:
        print(f"Errore nel caricamento dei dati: {e}")
        sys.exit(1)

    # Analizza la rete di collaborazioni
    logging.info("Inizio analisi della rete di collaborazioni.")
    print("Analisi della rete di collaborazioni...")
    try:
        G, metrics_df = analyze_network(collaborations)
    except Exception as e:
        logging.error(f"Errore nell'analisi della rete: {e}")
        print(f"Errore nell'analisi della rete: {e}")
        sys.exit(1)

    # Visualizza la rete
    visualize_network(G, metrics_df)

    # Analizza i repository
    logging.info("Inizio analisi dei repository.")
    print("Analisi dei repository...")
    try:
        analyze_repositories(repositories)
    except Exception as e:
        logging.error(f"Errore nell'analisi dei repository: {e}")
        print(f"Errore nell'analisi dei repository: {e}")

    # Analizza le Pull Requests
    logging.info("Inizio analisi delle Pull Requests.")
    print("Analisi delle Pull Requests...")
    try:
        analyze_pull_requests(pull_requests)
    except Exception as e:
        logging.error(f"Errore nell'analisi delle Pull Requests: {e}")
        print(f"Errore nell'analisi delle Pull Requests: {e}")

    # Calcola statistiche di rete aggiuntive
    calculate_network_statistics(G)

    # Analizza metriche aggiuntive
    logging.info("Inizio analisi delle metriche aggiuntive.")
    print("Analisi delle metriche aggiuntive...")
    try:
        analyze_additional_metrics(G, metrics_df)
    except Exception as e:
        logging.error(f"Errore nell'analisi delle metriche aggiuntive: {e}")
        print(f"Errore nell'analisi delle metriche aggiuntive: {e}")

    # Visualizza grafici aggiuntivi e delle comunità
    visualize_additional_graphs(metrics_df)
    visualize_community_graph(G, metrics_df)

    # Chiamata all'analisi dinamica dell'assortatività
    dynamic_assortativity_over_time(collaborations)

    # Chiamata all'analisi dei contributori con sub-comunità
    developer_metrics_analysis_with_subcommunities(G, metrics_df)

    logging.info("Analisi completata con successo.")
    print("Analisi completata con successo.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Tempo totale di esecuzione: {elapsed_time:.2f} secondi.")
    print(f"Tempo totale di esecuzione: {elapsed_time:.2f} secondi.")
