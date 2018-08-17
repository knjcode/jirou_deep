#!/usr/bin/env python
# coding: utf-8

import os
import re
import tweepy
import checker

import schedule
import time

consumer_key    = os.environ['CONSUMER_KEY']
consumer_secret = os.environ['CONSUMER_SECRET']
access_token    = os.environ['ACCESS_TOKEN']
access_secret   = os.environ['ACCESS_SECRET']
bot_name        = os.environ['BOT_NAME']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

def on_status(status):
    # reply
    if status.in_reply_to_screen_name == bot_name:
        reply_url = "https://twitter.com/%s/status/%s" % (status.author.screen_name, status.id_str)
        try:
            print("reply from %s %s" % (status.author.screen_name, reply_url))
            print(status.text)
        except:
            pass

        try:
            media_url_https = status.entities['media'][0]['media_url_https']
            print("image found: %s" % media_url_https)
            results = checker.predict(media_url_https)
            print(results)

            top = results[0]['term']
            debug_arr = ["%s (%2.2f%%)" % (result['term'], result['score'] * 100) for result in results[:3]]

            tweet = "@%s %s です\n--- top3 ---\n%s\n %s" % \
                (status.user.screen_name, top, '\n'.join(debug_arr), reply_url)

            api.update_status(status=tweet, in_reply_to_status_id=status.id)
        except:
            pass

    return True


def load_latest_reply_id():
    try:
        with open('model/last_reply_id') as f:
            last_seen = int(f.read().strip())
    except IOError:
        # If last_reply_id file does not exist, use the bot's latest tweet ID
        last_seen = api.me().status.id
        save_latest_reply_id(last_seen)
    return last_seen


def save_latest_reply_id(id):
    with open('model/last_reply_id', 'w') as f:
        f.write(str(id))
    print("write last_reply_id: %s" % id)


def check_mentions_timeline():
    last_seen = load_latest_reply_id()
    last_id = 0
    for status in api.mentions_timeline(since_id=last_seen):
        on_status(status)
        if status.id > last_id:
            last_id = status.id
            save_latest_reply_id(last_id)


interval_in_seconds = max(int(os.environ['INTERVAL_IN_SECONDS']), 12)
schedule.every(interval_in_seconds).seconds.do(check_mentions_timeline)

while True:
    schedule.run_pending()
    time.sleep(1)
