import pypyodbc
import yagmail
from twilio.rest import Client
import keyring
import json
import requests
import urllib
import smtplib
from email.mime.text import MIMEText
import telepot
import telepot.loop
import praw
import os
import threading
import time


class General:
    max_verbosity = 10

    def __init__(self, verbosity: int = max_verbosity):
        """
        Initializes the verbosity of the printer.

        :param verbosity: The level of the amount of printing of the program.
        High verbosity means lots of printing
        """
        self.verbosity = verbosity

    def pprint(self, level: int, *values):
        """
        Basically the same as print. Important functions must have high levels

        :param level: The threshold that needs to be less than or equal to verbosity
        :param values: The things to print
        """
        if abs(self.max_verbosity - level + 1) < self.verbosity:
            print(*values)

    @staticmethod
    def sanitise(text: str):
        """
        Sanitises input and removes possible SQL injections.

        :param text: The text to sanitise
        :return: The sanitised text
        """
        # Removes new lines, weird characters and dialogue
        text = " " + text + " "

        lined_text = text.split("\n")
        text = ""
        # Remove dialogue
        for line in lined_text:
            if ":" in line:
                if line.index(":") < 15:
                    index = line.index(":") + 1
                else:
                    index = 0
            else:
                index = 0
            text = text + "\n" + line[index:]

        # Lower case everything
        text = text.lower()

        text = text.replace("'s", " is")
        text = text.replace("'ve", " have")
        text = text.replace("n't", " not")
        text = text.replace("I'm", "I am")
        text = text.replace("'re", " are")
        text = text.replace("’s", " is")
        text = text.replace("’ve", " have")
        text = text.replace("n’t", " not")
        text = text.replace("I’m", "I am")
        text = text.replace("’re", " are")

        # Remove weird characters and double spaces
        weird_characters = [".", ",", "?", "!", "'", "’", "\"", "\n", "\t", "-", "/", "[", "]", "(", ")", ":", "“", "”"]
        for weird_character in weird_characters:
            text = text.replace(weird_character, " ")

        while "  " in text:
            text = text.replace("  ", " ")

        return text

    def simplify(self, text: str):
        """
        Takes a sentence and tries to simplify it possibly to aid a bag of words model.

        :param text: The text to simplify
        :return: The simplified text
        """
        # sanitise the input, remove common words and swap out similar words
        text = self.sanitise(text)
        # Decided not to use stopwords because that would take out too many and then get false positives
        common_words = ["what", "i", "you", "do", "a", "thing", "of", "was", "would", "were", "are", "been"
                                                                                                     "reddit", "why",
                        "your", "it", "is", "the", "that", "has", "had", "for", "at", "in", "on",
                        "with", "if", "to", "be", "and", "some"]
        for common_word in common_words:
            text = text.replace(" " + common_word + " ", " ")

        # Replace common synonyms
        synonyms = [["funniest", "funny", "hilarious"], ["food", "diet"], ["life", "lifetime"],
                    ["stop", "end", "quit"], ["advice", "help"],
                    ["grossest", "worst"], ["buy", "bought", "purchase"],
                    ["scary", "scariest", "spookiest"]]
        for synonym in synonyms:
            for word in synonym:
                text = text.replace(" " + word + " ", " " + synonym[0] + " ")

        # Stem words
        from nltk.stem.porter import PorterStemmer
        ps = PorterStemmer()
        text = text.split()
        text = [ps.stem(word) for word in text]
        text = " ".join(text)
        text = " " + text + " "

        # Missed words that still need to be stemmed:
        text = text.replace(" given ", " give ")
        text = text.replace(" younger ", " young ")

        return text

    @staticmethod
    def sleep_until(hour, minute=0):
        """
        Sleeps the script until that hour of the day which is either tomorrow or today.

        :param minute: The time on the minute to start running again
        :param hour: The time on the hour to start running again
        :return:
        """
        import datetime
        from time import sleep

        now = datetime.datetime.now()
        to = (now + datetime.timedelta(days=1)).replace(hour=hour, minute=minute, second=0)
        duration = (to - now).seconds
        sleep(duration)


class Email:
    @staticmethod
    def send_email(receiver_email: str, subject: str, message_text: str, username="christopher1duplessis"):
        """
        Sends an email using the given parameters.

        :param receiver_email: The email of the person to receive the message
        :param subject: The subject line of the email
        :param message_text: The actual email message
        :param username: The sender email
        """
        yag = yagmail.SMTP(username, keyring.get_password("email", username))
        yag.send(
            to=receiver_email,
            subject=subject,
            contents=message_text)

    @staticmethod
    def send_email2(receiver_email: str, subject: str, message: str, username="christopher1duplessis"):
        """
        Sends an email using the given parameters.

        :param receiver_email: The email of the person to receive the message
        :param subject: The subject line of the email
        :param message: The actual email message
        :param username: The sender email
        """
        smtp_ssl_host = 'smtp.gmail.com'  # smtp.mail.yahoo.com
        smtp_ssl_port = 465
        username = username + '@gmail.com'
        password = keyring.get_password("email", username)
        sender = username
        targets = [receiver_email]

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(targets)

        server = smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port)
        server.login(username, password)
        server.sendmail(sender, targets, msg.as_string())
        server.quit()


class SMS:
    @staticmethod
    def send_sms(message_text: str, receiver_sms="+27749911999"):
        """
        Sends an SMS using a given message text and receiver number.

        :param message_text: The text message
        :param receiver_sms: The sms number of the person to receive the SMS
        """
        user = "AC6925977501b11f3f5ea71105df8a4ea7"
        twilio_client = Client(user, keyring.get_password("twilio", user))
        twilio_client.messages.create(to=receiver_sms,
                                      from_="+19149964656",
                                      body=message_text)


class Telegram:
    # See https://core.telegram.org/bots/api#available-types
    TOKEN = keyring.get_password("telegram", "bot")
    URL = "https://api.telegram.org/bot{}/".format(TOKEN)

    def __init__(self):
        self.bot = telepot.Bot(self.TOKEN)

    def send_message(self, chat_id, text):
        self.bot.sendMessage(chat_id, text)

    def send_photo(self, chat_id, photo, caption=None):
        # bot.sendPhoto(chat_id, photo=open('path', 'rb'))
        # bot.sendPhoto(chat_id, 'your URl')
        self.bot.sendPhoto(chat_id, photo, caption)

    def create_daemon(self, handle, callback_handle=None):
        """
        Creates a daemon thread to keep running for ever.

        :param callback_handle: The function that takes a telegram callback query
        and tells the bot what to do in response.
        :param handle: The function that takes a telepot msg object
        and tells the bot what to do in response.
        """
        if callback_handle is None:
            telepot.loop.MessageLoop(self.bot, handle).run_as_thread()
        else:
            telepot.loop.MessageLoop(self.bot, {'chat': handle,
                                                'callback_query': callback_handle}).run_as_thread()


class TelegramDepreciated:
    # See https://core.telegram.org/bots/api#available-types
    TOKEN = keyring.get_password("telegram", "bot")
    URL = "https://api.telegram.org/bot{}/".format(TOKEN)

    @staticmethod
    def get_url(url):
        response = requests.get(url)
        content = response.content.decode("utf8")
        return content

    def get_json_from_url(self, url):
        content = self.get_url(url)
        js = json.loads(content)
        return js

    @staticmethod
    def get_last_update_id(updates):
        update_ids = []
        for update in updates["result"]:
            update_ids.append(int(update["update_id"]))
        return max(update_ids)

    def get_updates(self, offset=None):
        url = self.URL + "getUpdates?timeout=100"
        if offset:
            url += "&offset={}".format(offset)
        js = self.get_json_from_url(url)
        return js

    @staticmethod
    def get_last_chat_id_and_text(updates):
        num_updates = len(updates["result"])
        last_update = num_updates - 1
        text = updates["result"][last_update]["message"]["text"]
        chat_id = updates["result"][last_update]["message"]["chat"]["id"]
        return text, chat_id

    def send_message(self, text, chat_id):
        text = urllib.parse.quote_plus(text)
        url = self.URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
        self.get_url(url)


class RedditBot:
    # See https://praw.readthedocs.io/en/latest/code_overview/reddit_instance.html
    def __init__(self):
        """
        Perform all the main initialization
        """
        self.reddit = praw.Reddit('bot1')
        self.thread = None
        self.handle = None
        self.refresh_delay = None
        # Manage the checked comments
        if not os.path.isfile("reddit_comments_replied_to.txt"):
            self.checked_comments = []
        else:
            # Read the file into a list and remove any empty values
            with open("reddit_comments_replied_to.txt", "r") as f:
                self.checked_comments = f.read()
                self.checked_comments = self.checked_comments.split("\n")
                self.checked_comments = list(filter(None, self.checked_comments))
        # Manage the checked posts
        if not os.path.isfile("reddit_posts_replied_to.txt"):
            self.checked_posts = []
        else:
            with open("reddit_posts_replied_to.txt", "r") as f:
                self.checked_posts = f.read()
                self.checked_posts = self.checked_posts.split("\n")
                self.checked_posts = list(filter(None, self.checked_posts))

    def get_new_messages(self):
        """
        Gets new messages sent to the bot. inbox is a list of Message and Comment
        Message and Comment has the following attributes:
        body id subreddit author
        And the following functions:
        reply(text)

        Mark it as read with
        reddit.inbox.mark_read([item]) where item is an element of inbox
        """
        inbox = list(self.reddit.inbox.unread(limit=10))
        inbox.reverse()
        return inbox

    def get_submissions(self, subreddit_str: str, new_or_hot="new", limit=5):
        """
        submission.is_self is whether the submission is a text post.
        Attributes:
        id author title selftext.

        :param subreddit_str: The subreddit without r/ to get submissions
        :param new_or_hot: whether it should get new or hot submissions
        :param limit: The total number of submissions to get
        :return:
        """
        subreddit = self.reddit.subreddit(subreddit_str)
        if "new" in new_or_hot:
            return subreddit.new(limit=limit)
        elif "hot" in new_or_hot:
            return subreddit.hot(limit=limit)
        else:
            return subreddit.top(limit=limit)

    def update_posts_replied_to(self, post_id):
        """
        Updates the text file with the latest id.

        :param post_id: A string of the post id
        """
        self.checked_posts.append(post_id)
        with open("reddit_posts_replied_to.txt", "w") as f:
            for post in self.checked_posts:
                f.write(post + "\n")

    def update_comments_replied_to(self, comment_id):
        """
        Updates the text file with the latest id.

        :param comment_id: A string of the comment id
        """
        self.checked_comments.append(comment_id)
        with open("reddit_comments_replied_to.txt", "w") as f:
            for comment in self.checked_comments:
                f.write(comment + "\n")

    def create_daemon(self, handle, refresh_delay=5):
        """
        Creates an accessible daemon thread to keep running for ever.

        :param refresh_delay: The delay in seconds before loading the inbox
        :param handle: The function that takes a Comment or Message object
        and tells the bot what to do in response.
        """
        self.handle = handle
        self.refresh_delay = refresh_delay
        self.thread = threading.Thread(name="Reddit Daemon", target=self._keep_getting_new_messages)
        self.thread.daemon = True
        self.thread.start()

    def _keep_getting_new_messages(self):
        """
        The function that is called by the Daemon and needs
        self.handle and self.refresh_delay
        """
        while True:
            new_messages = self.get_new_messages()
            for message in new_messages:
                self.handle(message)
            time.sleep(self.refresh_delay)


class Google:
    def __init__(self, verbosity=5):
        self.verbosity = verbosity

    def search(self, query: str):
        """
        Performs a Google search with the given query and returns the highest ranked link.
        Returns None if it cannot find one or if something went wrong.

        :param query: The string of the query
        :return: The str of the url result
        """
        from googlesearch import search
        from urllib.error import HTTPError
        search_successful = False
        result = None

        # top level domains for the google search
        tld_array = ["com", "co.in", "co.za", "co.uk", "co.de", "co.id"]
        # the index of the top level domains to start off with
        tld_index = 0

        # if getting too many requests, change tld to co.in and com, co.za
        while not search_successful:
            try:
                urls = search(query, tld=tld_array[tld_index], num=1, stop=1, pause=2,
                              # domains=[""],
                              user_agent="GoogleSearchBotThing/1.0")
                for url in urls:
                    result = url

                search_successful = True
            except HTTPError as error:
                tld_index = (tld_index + 1) % len(tld_array)
                printer = General(self.verbosity)
                printer.pprint(8, "Too many requests from TLD. Switching to", tld_array[tld_index], error)
                search_successful = False
                pass
        return result


class TextSpeech:
    def __init__(self):
        self.response = None

    def get_text_to_speech(self, message_text: str):
        """
        Connects to Google servers and gets the audio recorded version of the message text.

        :param message_text: The text to be converted to speech
        """
        from google.cloud import texttospeech

        # Instantiates a client
        client = texttospeech.TextToSpeechClient()

        # Set the text input to be synthesized
        synthesis_input = texttospeech.types.SynthesisInput(text=message_text)

        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        # https://cloud.google.com/text-to-speech/docs/voices
        '''
        Favourites:
        WaveNet	en-IN	en-IN-Wavenet-C	MALE
        WaveNet	en-GB	en-GB-Wavenet-B	MALE
        '''
        voice = texttospeech.types.VoiceSelectionParams(
            language_code='en-US',
            name="en-GB-Wavenet-B",
            ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

        # Select the type of audio file you want returned
        audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.MP3)

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        self.response = client.synthesize_speech(synthesis_input, voice, audio_config)

    def save_response_as_mp3(self, mp3_name: str = "output.mp3"):
        """
        Saves the response voice note from Google as an mp3 file.
        Note that get_text_to_speech must have been run before calling this.

        :param mp3_name: The name of the mp3 file to be saved as
        """
        # The response's audio_content is binary.
        with open(mp3_name, 'wb') as out:
            # Write the response to the output file.
            out.write(self.response.audio_content)
            print('Audio content written to file "output.mp3"')

    @staticmethod
    def list_voices():
        """
        Lists the available voices from Google.
        """
        from google.cloud import texttospeech
        from google.cloud.texttospeech import enums
        client = texttospeech.TextToSpeechClient()

        # Performs the list voices request
        voices = client.list_voices()

        for voice in voices.voices:
            # Display the voice's name. Example: tpc-vocoded
            print('Name: {}'.format(voice.name))

            # Display the supported language codes for this voice. Example: "en-US"
            for language_code in voice.language_codes:
                print('Supported language: {}'.format(language_code))

            ssml_gender = enums.SsmlVoiceGender(voice.ssml_gender)

            # Display the SSML Voice Gender
            print('SSML Voice Gender: {}'.format(ssml_gender.name))

            # Display the natural sample rate hertz for this voice. Example: 24000
            print('Natural Sample Rate Hertz: {}\n'.format(
                voice.natural_sample_rate_hertz))


class Database:
    def __init__(self, database_file_name: str):
        """
        :param database_file_name: A relative or absolute path to the database

        For example:

        'C:\\Users\\Bob\\Documents\\PycharmProjects\\ProjectsToKeepRunning\\Emailing\\Loadshedding.accdb'

        OR

        'Emailing\\Loadshedding.accdb'
        """
        if os.path.isabs(database_file_name):
            self.file_name = database_file_name
        else:
            self.file_name = os.path.join(os.getcwd(), "") + database_file_name

    def select(self, statement: str, headings):
        """
        Executes a select statement. Returns a dictionary of headings which are arrays of entries.
        Warning: Access uses ? and * as wildcards while pyODBC uses _ and %

        :param statement: The SQL SELECT statement to be run
        :param headings: The names of the headings to return. Must be array-like
        """
        pypyodbc.lowercase = False
        conn = pypyodbc.connect(
            r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};" +
            r"Dbq=" + self.file_name)
        cur = conn.cursor()

        cur.execute(statement)

        result = {}
        for heading in headings:
            result[heading] = []

        while True:
            row = cur.fetchone()
            if row is None:
                break
            for heading in headings:
                result[heading].append(row.get(heading))

        cur.close()
        conn.close()

        return result

    def update_insert(self, statement):
        """
        Executes an update or an insert statement.

        :param statement: The SQL statement to execute
        """
        pypyodbc.lowercase = False
        conn = pypyodbc.connect(
            r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};" +
            r"Dbq=" + self.file_name)
        cur = conn.cursor()
        cur.execute(statement)
        conn.commit()

        cur.close()
        conn.close()
