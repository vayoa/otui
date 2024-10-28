# %%
import pandas as pd
from pybooru import Danbooru
from rich import print
import requests
from io import BytesIO
from PIL import Image

CLIENT = Danbooru("danbooru")


def get_posts(tags: str, page: int = 0, limit: int = 100) -> pd.DataFrame:
    posts = pd.DataFrame(
        CLIENT.post_list(tags=tags, page=page, limit=limit),
        columns=[
            "id",
            "score",
            "tag_string_general",
            "tag_string_character",
            "tag_string_copyright",
            "tag_string_artist",
            "file_url",
            "large_file_url",
            "preview_file_url",
        ],
    )
    posts["tags"] = (
        posts["tag_string_general"]
        + " "
        + posts["tag_string_character"]
        + " "
        + posts["tag_string_copyright"]
        + " "
        + posts["tag_string_artist"]
    )
    posts["tags_set"] = posts["tags"].apply(lambda x: set(x.split()))
    return posts.drop(
        columns=[
            "tag_string_general",
            "tag_string_character",
            "tag_string_copyright",
            "tag_string_artist",
        ]
    )


def search(tags: str, max_limit: int = 200, step: int = 100):
    tags_list = tags.replace(",", " ").split()
    tags_set = set(tags_list)
    two_tags = " ".join(tags_list[:2])
    page = 0
    all_posts = pd.DataFrame()
    while page * step <= max_limit:
        posts = get_posts(two_tags, page=page, limit=step)
        page += 1
        sim = posts.apply(
            lambda x: len(x["tags_set"] & tags_set) / float(len(tags_set)) * 100,
            axis=1,  # Correct axis argument
        )
        posts["similarity"] = sim
        all_posts = pd.concat([posts, all_posts])

        # Check if any of the individual tags are in the 'tags' column
        if sum(posts["similarity"] >= 90) > 0:
            break

    return all_posts.sort_values(by="similarity", ascending=False)


# %%
posts = search(
    "pocahontas_\(disney\) wearing a halloween_costume revealing_clothes, narrowed_eyes, huge_smile, looking_at_penis, dark-skinned_female, caressing_testicles, testicle_grab, 1girl, 1boy, huge dark-skinned_male bald man, smegma, huge_penis, speech_bubble",
    max_limit=1000,
)
images = [
    Image.open(BytesIO(requests.get(url).content))
    for url in posts[posts["similarity"] >= 90]["large_file_url"].iloc[:4]
]

for image in images:
    image.show()

# %%
