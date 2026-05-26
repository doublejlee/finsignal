import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_youtube_client():
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY not found — check your .env file")
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)

def get_recent_videos(channel_id: str, max_results: int = 5) -> list:
    """Fetch most recent video IDs from a channel."""
    youtube = get_youtube_client()
    
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results,
        order="date",
        type="video"
    )
    response = request.execute()
    
    videos = []
    for item in response["items"]:
        videos.append({
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
            "channel_id": channel_id,
            "channel_name": item["snippet"]["channelTitle"]
        })
    
    return videos

if __name__ == "__main__":
    # Test with Meet Kevin's channel
    videos = get_recent_videos("UCGy7SkBjcIAgTiwkXEtPnYg", max_results=3)
    for v in videos:
        print(f"{v['title'][:60]} — {v['video_id']}")