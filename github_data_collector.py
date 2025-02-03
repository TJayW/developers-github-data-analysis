import os
import sys
import time
import logging
import pandas as pd
from tqdm import tqdm
from github import Github
from github.GithubException import RateLimitExceededException, GithubException

# Configura il logging
logging.basicConfig(
    filename='script_log.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Recupera il token da una variabile d'ambiente
GITHUB_TOKEN = 'YOUR_GIT_HUB_KEY'

if not GITHUB_TOKEN:
    logging.error("Il token GitHub non è stato trovato nelle variabili d'ambiente.")
    sys.exit("Errore: Il token GitHub non è stato trovato nelle variabili d'ambiente.")

# Inizializza l'oggetto GitHub
g = Github(GITHUB_TOKEN, per_page=50, retry=3, timeout=15)

def check_rate_limit(g):
    """Controlla il rate limit e attende se necessario."""
    try:
        core_rate_limit = g.get_rate_limit().core
        remaining = core_rate_limit.remaining
        reset_timestamp = core_rate_limit.reset.timestamp()
        current_timestamp = time.time()
        if remaining < 10:
            sleep_time = reset_timestamp - current_timestamp + 10  # Aggiungi 10 secondi di buffer
            if sleep_time > 0:
                logging.warning(f"Rate limit raggiunto. Attendo per {sleep_time/60:.2f} minuti.")
                print(f"Rate limit raggiunto. Attendo per {sleep_time/60:.2f} minuti.")
                time.sleep(sleep_time)
    except GithubException as e:
        logging.error(f"Errore nel controllo del rate limit: {e}")
        time.sleep(60)  # Attendi un minuto prima di riprovare

def get_target_repositories(topic, num_repos):
    """
    Ottiene i repository target basati su topic e numero di stelle.
    Filtra repository non fork e non archiviati.
    """
    query = f"topic:{topic} stars:>100 fork:false archived:false"
    repositories = []
    
    logging.info(f"Inizio ricerca dei repository con query: '{query}'")
    print("Ottenimento dei repository target...")
    
    try:
        result = g.search_repositories(query=query, sort='stars', order='desc')
        for repo in tqdm(result, total=num_repos, desc="Ottenimento repository"):
            repositories.append(repo.full_name)
            if len(repositories) >= num_repos:
                break
    except RateLimitExceededException:
        check_rate_limit(g)
        return get_target_repositories(topic, num_repos)
    except GithubException as e:
        logging.error(f"Errore durante la ricerca dei repository: {e}")
        return repositories

    logging.info(f"Repository selezionati: {len(repositories)}")
    return repositories

def get_top_contributors(repo_full_name, num_contributors):
    """
    Ottiene i principali contributori di un repository.
    """
    contributors = []
    try:
        repo = g.get_repo(repo_full_name)
        contributors_api = repo.get_contributors()
        for contributor in contributors_api:
            contributors.append(contributor.login)
            if len(contributors) >= num_contributors:
                break
    except RateLimitExceededException:
        check_rate_limit(g)
        return get_top_contributors(repo_full_name, num_contributors)
    except GithubException as e:
        logging.error(f"Errore ottenendo i contributori per {repo_full_name}: {e}")
    return contributors

def get_developer_details(username):
    """
    Ottiene i dettagli di uno sviluppatore.
    """
    try:
        user = g.get_user(username)
        return {
            'login': user.login,
            'id': user.id,
            'name': user.name,
            'company': user.company,
            'location': user.location,
            'email': user.email,
            'public_repos': user.public_repos,
            'followers': user.followers,
            'following': user.following,
            'bio': user.bio,
            'created_at': user.created_at.isoformat(),
            'updated_at': user.updated_at.isoformat(),
            'public_gists': user.public_gists,
            'hireable': user.hireable
        }
    except GithubException as e:
        logging.error(f"Errore ottenendo dettagli per l'utente {username}: {e}")
        return {}

def get_repository_details(repo_full_name):
    """
    Ottiene i dettagli di un repository.
    """
    try:
        repo = g.get_repo(repo_full_name)
        return {
            'full_name': repo.full_name,
            'description': repo.description,
            'created_at': repo.created_at.isoformat(),
            'updated_at': repo.updated_at.isoformat(),
            'pushed_at': repo.pushed_at.isoformat(),
            'language': repo.language,
            'stargazers_count': repo.stargazers_count,
            'forks_count': repo.forks_count,
            'open_issues_count': repo.open_issues_count,
            'watchers_count': repo.watchers_count,
            'size': repo.size,
            'default_branch': repo.default_branch,
            'license': repo.license.name if repo.license else None,
            'topics': repo.get_topics()
        }
    except GithubException as e:
        logging.error(f"Errore ottenendo dettagli per il repository {repo_full_name}: {e}")
        return {}

def get_pull_request_details(repo_full_name, pr_number):
    """
    Ottiene i dettagli di una pull request.
    """
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        return {
            'pr_number': pr.number,
            'title': pr.title,
            'state': pr.state,
            'merged': pr.merged,
            'mergeable': pr.mergeable,
            'created_at': pr.created_at.isoformat(),
            'closed_at': pr.closed_at.isoformat() if pr.closed_at else None,
            'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
            'author': pr.user.login if pr.user else None,
            'base_branch': pr.base.ref,
            'head_branch': pr.head.ref,
            'changed_files': pr.changed_files,
            'additions': pr.additions,
            'deletions': pr.deletions,
            'comments_count': pr.comments,
            'reviews_count': pr.review_comments
        }
    except GithubException as e:
        logging.error(f"Errore ottenendo dettagli per PR #{pr_number} in {repo_full_name}: {e}")
        return {}

def get_collaborations(repo_full_name, contributors, num_prs=100):
    """
    Ottiene le collaborazioni tra i contributori basate sui pull request.
    Aggiunge timestamp e tipo di interazione.
    """
    edges = []
    try:
        repo = g.get_repo(repo_full_name)
        pulls = repo.get_pulls(state='closed', sort='updated', direction='desc')
        pr_count = 0

        for pr in pulls:
            if pr.merged_at is None:
                continue
            pr_count += 1
            if pr_count > num_prs:
                break

            author = pr.user.login if pr.user else None
            if not author or author not in contributors:
                continue

            # Revisioni
            try:
                reviews = pr.get_reviews()
                for review in reviews:
                    reviewer = review.user.login if review.user else None
                    if reviewer and reviewer in contributors and reviewer != author:
                        edges.append({
                            'source': author,
                            'target': reviewer,
                            'repository': repo_full_name,
                            'interaction_type': 'review',
                            'timestamp': review.submitted_at.isoformat() if review.submitted_at else None
                        })
            except GithubException as e:
                logging.error(f"Errore ottenendo le revisioni per PR #{pr.number} in {repo_full_name}: {e}")

            # Commenti
            try:
                comments = pr.get_comments()
                for comment in comments:
                    commenter = comment.user.login if comment.user else None
                    if commenter and commenter in contributors and commenter != author:
                        edges.append({
                            'source': author,
                            'target': commenter,
                            'repository': repo_full_name,
                            'interaction_type': 'comment',
                            'timestamp': comment.created_at.isoformat() if comment.created_at else None
                        })
            except GithubException as e:
                logging.error(f"Errore ottenendo i commenti per PR #{pr.number} in {repo_full_name}: {e}")

            # Check rate limit periodically
            if pr_count % 50 == 0:
                check_rate_limit(g)

    except RateLimitExceededException:
        check_rate_limit(g)
        return get_collaborations(repo_full_name, contributors, num_prs)
    except GithubException as e:
        logging.error(f"Errore ottenendo collaborazioni per {repo_full_name}: {e}")
    return edges

def save_developer_details(developer_details, filename='developers_large.csv'):
    """
    Salva i dettagli degli sviluppatori in un file CSV.
    """
    df = pd.DataFrame(developer_details)
    if not df.empty:
        # Verifica se il file esiste per decidere se scrivere l'header
        write_header = not os.path.exists(filename)
        df.to_csv(filename, mode='a', header=write_header, index=False, sep=',', quotechar='"', quoting=1)
        logging.info(f"Salvati {len(df)} sviluppatori in '{filename}'")
    else:
        logging.info("Nessun dettaglio sviluppatore da salvare.")

def save_repository_details(repository_details, filename='repositories_large.csv'):
    """
    Salva i dettagli dei repository in un file CSV.
    """
    df = pd.DataFrame(repository_details)
    if not df.empty:
        write_header = not os.path.exists(filename)
        df.to_csv(filename, mode='a', header=write_header, index=False, sep=',', quotechar='"', quoting=1)
        logging.info(f"Salvati {len(df)} repository in '{filename}'")
    else:
        logging.info("Nessun dettaglio repository da salvare.")

def save_collaborations(collaborations, filename='collaborations_large.csv'):
    """
    Salva le collaborazioni in un file CSV.
    """
    df = pd.DataFrame(collaborations)
    if not df.empty:
        write_header = not os.path.exists(filename)
        df.to_csv(filename, mode='a', header=write_header, index=False, sep=',', quotechar='"', quoting=1)
        logging.info(f"Salvate {len(df)} collaborazioni in '{filename}'")
    else:
        logging.info("Nessuna collaborazione da salvare.")

def save_pull_requests(pull_requests, filename='pull_requests_large.csv'):
    """
    Salva i dettagli delle pull requests in un file CSV.
    """
    df = pd.DataFrame(pull_requests)
    if not df.empty:
        write_header = not os.path.exists(filename)
        df.to_csv(filename, mode='a', header=write_header, index=False, sep=',', quotechar='"', quoting=1)
        logging.info(f"Salvate {len(df)} pull requests in '{filename}'")
    else:
        logging.info("Nessuna pull request da salvare.")

def main():
    topic = 'machine-learning'  # Cambia il topic se necessario
    num_repos = 100             # Numero di repository da analizzare
    num_contributors = 50       # Numero di contributori per repository

    logging.info("Inizio il processo principale.")
    repositories = get_target_repositories(topic, num_repos)
    print(f"Repository selezionati: {len(repositories)}")
    logging.info(f"Repository selezionati: {len(repositories)}")

    all_contributors = set()
    repo_contributors = {}

    print("Ottenimento dei contributori per ogni repository...")
    logging.info("Inizio l'ottenimento dei contributori per ogni repository.")
    for repo in tqdm(repositories, desc="Ottenimento contributori"):
        contributors = get_top_contributors(repo, num_contributors)
        repo_contributors[repo] = contributors
        all_contributors.update(contributors)
        check_rate_limit(g)
        time.sleep(0.1)

    print(f"Totale contributori unici: {len(all_contributors)}")
    logging.info(f"Totale contributori unici: {len(all_contributors)}")

    # Raccogliere dettagli sugli sviluppatori
    print("Raccolta dei dettagli degli sviluppatori...")
    logging.info("Inizio la raccolta dei dettagli degli sviluppatori.")
    developer_details = []
    for developer in tqdm(all_contributors, desc="Raccolta dettagli sviluppatori"):
        details = get_developer_details(developer)
        if details:
            developer_details.append(details)
        check_rate_limit(g)
        time.sleep(0.1)

    save_developer_details(developer_details)
    print("Dettagli degli sviluppatori salvati in 'developers_large.csv'")
    logging.info("Dettagli degli sviluppatori salvati in 'developers_large.csv'")

    # Raccogliere dettagli sui repository
    print("Raccolta dei dettagli dei repository...")
    logging.info("Inizio la raccolta dei dettagli dei repository.")
    repository_details = []
    for repo in tqdm(repositories, desc="Raccolta dettagli repository"):
        details = get_repository_details(repo)
        if details:
            repository_details.append(details)
        check_rate_limit(g)
        time.sleep(0.1)

    save_repository_details(repository_details)
    print("Dettagli dei repository salvati in 'repositories_large.csv'")
    logging.info("Dettagli dei repository salvati in 'repositories_large.csv'")

    # Creare il dataset delle collaborazioni
    all_collaborations = []
    print("Ottenimento delle collaborazioni tra i contributori...")
    logging.info("Inizio l'ottenimento delle collaborazioni tra i contributori.")
    for repo in tqdm(repositories, desc="Ottenimento collaborazioni"):
        contributors = repo_contributors.get(repo, [])
        collaborations = get_collaborations(repo, contributors)
        if collaborations:
            all_collaborations.extend(collaborations)
        check_rate_limit(g)
        time.sleep(0.1)

    # Rimuovere duplicati
    unique_collaborations = [dict(t) for t in {tuple(sorted(d.items())) for d in all_collaborations}]

    save_collaborations(unique_collaborations)
    print("Collaborazioni salvate in 'collaborations_large.csv'")
    logging.info("Collaborazioni salvate in 'collaborations_large.csv'")

    # Raccogliere e salvare dettagli delle Pull Requests
    all_pull_requests = []
    print("Raccolta dei dettagli delle Pull Requests...")
    logging.info("Inizio la raccolta dei dettagli delle Pull Requests.")
    for repo in tqdm(repositories, desc="Raccolta dettagli PR"):
        try:
            repo_obj = g.get_repo(repo)
            pulls = repo_obj.get_pulls(state='closed', sort='updated', direction='desc')
            pr_count = 0
            for pr in pulls:
                if pr.merged_at is None:
                    continue  # Considera solo i PR fusi
                pr_count += 1
                if pr_count > 100:
                    break
                pr_details = get_pull_request_details(repo, pr.number)
                if pr_details:
                    all_pull_requests.append(pr_details)
                check_rate_limit(g)
                time.sleep(0.1)
        except RateLimitExceededException:
            check_rate_limit(g)
            continue
        except GithubException as e:
            logging.error(f"Errore ottenendo le PR per {repo}: {e}")
            continue

    save_pull_requests(all_pull_requests)
    print("Dettagli delle Pull Requests salvati in 'pull_requests_large.csv'")
    logging.info("Dettagli delle Pull Requests salvati in 'pull_requests_large.csv'")

if __name__ == "__main__":
    main()
