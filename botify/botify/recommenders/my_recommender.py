import random

from .contextual import Contextual
from .recommender import Recommender


class Foo(Recommender):
    def __init__(self, tracks_redis, artists_redis, catalog, recommendations_by_artist_neighbours,
                 recommendation_personal_top):
        self.fallback = Contextual(tracks_redis, catalog)
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.catalog = catalog
        self.recommendations_by_artist_neighbours = recommendations_by_artist_neighbours
        self.recommendation_personal_top = recommendation_personal_top

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        recommendations = self.recommendation_personal_top.get(user)

        if recommendations is not None and prev_track_time < 0.3:
            shuffled = list(self.catalog.from_bytes(recommendations))
            random.shuffle(shuffled)
            return shuffled[0]

        if prev_track_time >= 0.9:
            track_data = self.tracks_redis.get(prev_track)

            if track_data is not None:
                track = self.catalog.from_bytes(track_data)
            else:
                raise ValueError(f"Track not found: {prev_track}")

            artist_data = self.artists_redis.get(track.artist)

            if artist_data is not None:
                recommendations = self.recommendations_by_artist_neighbours.get(artist_data)

                if recommendations is not None:
                    shuffled = list(self.catalog.from_bytes(recommendations))
                    random.shuffle(shuffled)
                    return shuffled[0]
            else:
                raise ValueError(f"Artist not found: {prev_track}")

        return self.fallback.recommend_next(user, prev_track, prev_track_time)
