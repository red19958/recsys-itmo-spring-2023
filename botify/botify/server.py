import json
import logging
import time
from dataclasses import asdict
from datetime import datetime

from flask import Flask
from flask_redis import Redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer
from data import DataLogger, Datum
from experiment import Experiments, Treatment
from recommenders.sticky_artist import StickyArtist
from recommenders.random import Random
from recommenders.toppop import TopPop
from recommenders.indexed import Indexed
from recommenders.contextual import Contextual
from recommenders.my_recommender import Foo
from track import Catalog

import numpy as np

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
artists_redis = Redis(app, config_prefix="REDIS_ARTIST")
recommendations_ub_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_UB")
recommendations_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS")
recommendations_by_artist_neighbours_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_BY_ARTIST_NEIGHBOURS")
recommendations_personal_top_redis = Redis(app, config_prefix="REDIS_PERSONAL_TOP")
tracks_with_diverse_recs_redis = Redis(app, config_prefix="REDIS_TRACKS_WITH_DIVERSE_RECS")

data_logger = DataLogger(app)

catalog = Catalog(app).load(
    app.config["TRACKS_CATALOG"], app.config["TOP_TRACKS_CATALOG"], app.config["TRACKS_WITH_DIVERSE_RECS_CATALOG"]
)

catalog.upload_artists(artists_redis.connection)
catalog.upload_tracks(tracks_redis.connection, tracks_with_diverse_recs_redis.connection)
catalog.upload_recommendations(recommendations_redis.connection)
catalog.upload_recommendations(recommendations_ub_redis.connection, "RECOMMENDATIONS_UB_FILE_PATH")
catalog.upload_recommendations_neighbours(recommendations_by_artist_neighbours_redis.connection)
catalog.upload_recommendations(recommendations_personal_top_redis.connection, "PERSONAL_TOP_FILE_PATH")

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)


class Hello(Resource):
    def get(self):
        return {
            "status": "alive",
            "message": "welcome to botify, the best toy music recommender",
        }


class Track(Resource):
    def get(self, track: int):
        data = tracks_redis.connection.get(track)
        if data is not None:
            return asdict(catalog.from_bytes(data))
        else:
            abort(404, description="Track not found")


class NextTrack(Resource):
    def post(self, user: int):
        start = time.time()
        args = parser.parse_args()

        treatment = Experiments.CHECK_HW.assign(user)

        if treatment == Treatment.T1:
            recommender = Foo(tracks_redis, artists_redis, catalog, recommendations_by_artist_neighbours_redis,
                              recommendations_personal_top_redis)
        else:
            recommender = Contextual(tracks_redis.connection, catalog)

        recommendation = recommender.recommend_next(user, args.track, args.time)

        data_logger.log(
            "next",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
                recommendation,
            ),
        )
        return {"user": user, "track": recommendation}


class LastTrack(Resource):
    def post(self, user: int):
        start = time.time()
        args = parser.parse_args()
        data_logger.log(
            "last",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
            ),
        )
        return {"user": user}


api.add_resource(Hello, "/")
api.add_resource(Track, "/track/<int:track>")
api.add_resource(NextTrack, "/next/<int:user>")
api.add_resource(LastTrack, "/last/<int:user>")

if __name__ == "__main__":
    http_server = WSGIServer(("", 5000), app)
    http_server.serve_forever()
