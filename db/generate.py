# %%
import requests
from rich import print
import time
import threading
import pandas as pd
from tqdm.auto import tqdm

CATEGORY_MAP = {
    0: "General",
    1: "Artist",
    3: "Copyright",
    4: "Character",
    5: "Meta",
}

# %%


def fetch_page(page=1, limit=2, post_count_ge_than=1):
    url = f"https://danbooru.donmai.us/tags.json?only=id,name,category,post_count,consequent_aliases[id,antecedent_name],wiki_page[body,other_names]&search[is_deprecated]=false&search[wiki_page][is_deleted]=false&search[post_count]=>={post_count_ge_than}&search[order]=count&limit={limit}&page={page}"
    response = requests.get(url).json()
    return response


def thread_requests(
    requests_per_second=10,
    initial_page=1,
    limit_per_request=1000,
    max_pages=15,
    post_count_ge_than=1,
) -> pd.DataFrame:
    """thread requests per second with the paging to get all the tags"""
    results = []
    page = initial_page

    with tqdm(total=max_pages * limit_per_request) as pbar:
        while page < initial_page + max_pages:
            start_time = time.time()
            threads = []
            for _ in range(requests_per_second):
                if page >= initial_page + max_pages:
                    break
                thread = threading.Thread(
                    target=lambda p=page, limit=limit_per_request, post_count_ge_than=post_count_ge_than: results.extend(
                        fetch_page(p, limit, post_count_ge_than)
                    )
                )
                threads.append(thread)
                thread.start()
                page += 1

            for thread in threads:
                thread.join()

            elapsed_time = time.time() - start_time
            if elapsed_time < 1:
                time.sleep(1 - elapsed_time)
            pbar.update(len(threads) * limit_per_request)

    df = pd.DataFrame(results)
    df["category"] = df["category"].replace(CATEGORY_MAP)
    df[["wiki_body", "wiki_other_names"]] = pd.json_normalize(df["wiki_page"])
    df = df.drop(columns=["wiki_page"])
    df["consequent_aliases"] = df["consequent_aliases"].apply(
        lambda x: [i["antecedent_name"] for i in x]
    )
    df["description"] = (
        df["wiki_body"]
        + "\naliases: "
        + df["consequent_aliases"].str.join(", ")
        + "\nother names: "
        + df["wiki_other_names"].str.join(", ")
    )
    df["document"] = df["name"] + " (" + df["category"] + "):\n" + df["description"]
    df = df.drop_duplicates(subset="id")
    return df


if __name__ == "__main__":
    df = thread_requests(max_pages=100, post_count_ge_than=30)
    df.to_parquet("danbooru_tags.parquet", index=False)
