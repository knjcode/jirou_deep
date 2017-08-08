#!/usr/bin/env python
# coding: utf-8

import os
import re
import tweepy
import checker

consumer_key    = os.environ['CONSUMER_KEY']
consumer_secret = os.environ['CONSUMER_SECRET']
access_token    = os.environ['ACCESS_TOKEN']
access_secret   = os.environ['ACCESS_SECRET']
bot_name        = os.environ['BOT_NAME']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

class myStreamListener(tweepy.StreamListener):
    def on_status(self, status):
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

                tweet = ".@%s %s です\n--- top3 ---\n%s\n %s" % \
                    (status.user.screen_name, top, '\n'.join(debug_arr), reply_url)

                api.update_status(status=tweet, in_reply_to_status_id=status.id)
            except:
                pass

        return True

    def on_error(self, status_code):
        print('An error has occured! Status code = %s' % status_code)
        return True

    def on_timeout(self):
        print('Timeout...')
        return True

myStreamListener = myStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
myStream.userstream()
