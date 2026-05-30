import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_youtube_client():
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY not found — check your .env file")
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)

def resolve_handle(handle: str) -> str:
    """Resolve a YouTube @handle (e.g. '@JosephCarlsonShow') to its channel ID."""
    youtube = get_youtube_client()
    request = youtube.channels().list(part="id", forHandle=handle.lstrip("@"))
    response = request.execute()
    items = response.get("items", [])
    if not items:
        raise ValueError(f"Could not resolve handle '{handle}' to a channel ID")
    return items[0]["id"]

def get_recent_videos(channel_id: str, max_results: int = 5, published_before: str = None) -> list:
    """Fetch the most recent videos from a channel, newest first.

    If `published_before` (RFC 3339, e.g. '2026-04-29T00:00:00Z') is given, only videos
    published before that instant are returned — used to pull history old enough to clear
    the backtest's 30-day horizon, so frequent posters yield evaluable calls instead of
    only un-scorable last-week uploads.

    Walks the channel's uploads playlist (reverse-chronological, complete, 1 quota unit per
    page) rather than search.list, which is non-exhaustive with `publishedBefore` — it
    silently returned 0 results for high-volume channels like Meet Kevin — and costs 100
    units per call.
    """
    youtube = get_youtube_client()

    ch = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    items = ch.get("items", [])
    if not items:
        raise ValueError(f"Could not find uploads playlist for channel '{channel_id}'")
    uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    videos = []
    page_token = None
    while len(videos) < max_results:
        response = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist,
            maxResults=50,
            pageToken=page_token,
        ).execute()

        for item in response["items"]:
            cd = item["contentDetails"]
            sn = item["snippet"]
            published_at = cd.get("videoPublishedAt") or sn["publishedAt"]
            # Uploads are newest-first, so a too-new video just gets skipped; once we pass
            # the cutoff every later item is older too.
            if published_before and published_at >= published_before:
                continue
            videos.append({
                "video_id": cd["videoId"],
                "title": sn["title"],
                "published_at": published_at,
                "channel_id": channel_id,
                "channel_name": sn.get("channelTitle", ""),
            })
            if len(videos) >= max_results:
                break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return videos

if __name__ == "__main__":
    # Test with Meet Kevin's channel
    videos = get_recent_videos("UCGy7SkBjcIAgTiwkXEtPnYg", max_results=3)
    for v in videos:
        print(f"{v['title'][:60]} — {v['video_id']}")