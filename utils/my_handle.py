import os
import threading
import json
import random
import difflib
import logging
from datetime import datetime
import traceback
import importlib
import pyautogui

from .config import Config
from .common import Common
from .audio import Audio
from .gpt_model.gpt import GPT_MODEL
from .logger import Configure_logger
from .db import SQLiteDB


"""
	___ _
	|_ _| | ____ _ _ __ ___  ___
	 | || |/ / _` | '__/ _ \/ __|
	 | ||   < (_| | | | (_) \__ \
	|___|_|\_\__,_|_|  \___/|___/

"""


class My_handle():
    common = None
    config = None
    audio = None

    def __init__(self, config_path):
        logging.info("初始化My_handle...")

        if My_handle.common is None:
            My_handle.common = Common()
        if My_handle.config is None:
            My_handle.config = Config(config_path)
        if My_handle.audio is None:
            My_handle.audio = Audio(config_path)

        # 日志文件路径
        file_path = "./log/log-" + My_handle.common.get_bj_time(1) + ".txt"
        Configure_logger(file_path)

        self.proxy = None
        # self.proxy = {
        #     "http": "http://127.0.0.1:10809",
        #     "https": "http://127.0.0.1:10809"
        # }

        try:
            # 数据丢弃部分相关的实现
            self.data_lock = threading.Lock()
            self.timers = {}

            # 设置会话初始值
            self.session_config = {'msg': [
                {"role": "system", "content": My_handle.config.get('chatgpt', 'preset')}]}
            self.sessions = {}
            self.current_key_index = 0

            # 直播间号
            self.room_id = My_handle.config.get("room_display_id")

            self.before_prompt = My_handle.config.get("before_prompt")
            self.after_prompt = My_handle.config.get("after_prompt")

            # 过滤配置
            self.filter_config = My_handle.config.get("filter")
            # 答谢
            self.thanks_config = My_handle.config.get("thanks")

            self.chat_type = My_handle.config.get("chat_type")

            self.need_lang = My_handle.config.get("need_lang")

            # 优先本地问答
            self.local_qa = My_handle.config.get("local_qa")
            self.local_qa_audio_list = None

            # 音频合成使用技术
            My_handle.audio_synthesis_type = My_handle.config.get(
                "audio_synthesis_type")

            # Stable Diffusion
            self.sd_config = My_handle.config.get("sd")

            # 点歌模块
            self.choose_song_config = My_handle.config.get("choose_song")
            self.choose_song_song_lists = None

            logging.info(f"配置数据加载成功。")
        except Exception as e:
            logging.error(traceback.format_exc())

        # 设置GPT_Model全局模型列表
        GPT_MODEL.set_model_config("openai", My_handle.config.get("openai"))
        GPT_MODEL.set_model_config("chatgpt", My_handle.config.get("chatgpt"))
        GPT_MODEL.set_model_config("claude", My_handle.config.get("claude"))

        """
        新增LLM后，这边先定义下各个变量，下面会用到
        """
        self.chatgpt = None
        self.claude = None
        self.claude2 = None
        self.chatglm = None
        self.chat_with_file = None
        self.text_generation_webui = None
        self.sparkdesk = None
        self.langchain_chatglm = None
        self.zhipu = None
        self.bard_api = None
        self.yiyan = None
        self.tongyi = None

        # 聊天相关类实例化
        if self.chat_type == "chatgpt":
            self.chatgpt = GPT_MODEL.get("chatgpt")

        elif self.chat_type == "claude":
            self.claude = GPT_MODEL.get(self.chat_type)

            # 初次运行 先重置下会话
            if not self.claude.reset_claude():
                logging.error("重置Claude会话失败喵~")
        elif self.chat_type == "claude2":
            GPT_MODEL.set_model_config(
                "claude2", My_handle.config.get("claude2"))

            self.claude2 = GPT_MODEL.get(self.chat_type)

            # 初次运行 先重置下会话
            if self.claude2.get_organization_id() is None:
                logging.error("重置Claude2会话失败喵~")
        elif self.chat_type == "chatterbot":
            from chatterbot import ChatBot  # 导入聊天机器人库

            self.chatterbot_config = My_handle.config.get("chatterbot")

            try:
                self.bot = ChatBot(
                    self.chatterbot_config["name"],  # 聊天机器人名字
                    database_uri='sqlite:///' + \
                    self.chatterbot_config["db_path"]  # 数据库URI，数据库用于存储对话历史
                )
            except Exception as e:
                logging.info(e)
                exit(0)
        elif self.chat_type == "chatglm":
            GPT_MODEL.set_model_config(
                "chatglm", My_handle.config.get("chatglm"))

            self.chatglm = GPT_MODEL.get(self.chat_type)
        elif self.chat_type == "chat_with_file":
            from utils.chat_with_file.chat_with_file import Chat_with_file
            self.chat_with_file = Chat_with_file(
                My_handle.config.get("chat_with_file"))
        elif self.chat_type == "text_generation_webui":
            GPT_MODEL.set_model_config(
                "text_generation_webui", My_handle.config.get("text_generation_webui"))

            self.text_generation_webui = GPT_MODEL.get(self.chat_type)
        elif self.chat_type == "sparkdesk":
            GPT_MODEL.set_model_config(
                "sparkdesk", My_handle.config.get("sparkdesk"))

            self.sparkdesk = GPT_MODEL.get(self.chat_type)
        elif self.chat_type == "langchain_chatglm":
            GPT_MODEL.set_model_config(
                "langchain_chatglm", My_handle.config.get("langchain_chatglm"))

            self.langchain_chatglm = GPT_MODEL.get(self.chat_type)
        elif self.chat_type == "zhipu":
            GPT_MODEL.set_model_config("zhipu", My_handle.config.get("zhipu"))

            self.zhipu = GPT_MODEL.get(self.chat_type)
        elif self.chat_type == "bard":
            GPT_MODEL.set_model_config("bard", My_handle.config.get("bard"))

            self.bard_api = GPT_MODEL.get(self.chat_type)
        elif self.chat_type == "yiyan":
            GPT_MODEL.set_model_config("yiyan", My_handle.config.get("yiyan"))

            self.yiyan = GPT_MODEL.get(self.chat_type)
        elif self.chat_type == "tongyi":
            GPT_MODEL.set_model_config(
                "tongyi", My_handle.config.get("tongyi"))

            self.tongyi = GPT_MODEL.get(self.chat_type)
        elif self.chat_type == "game":
            self.game = importlib.import_module(
                "game." + My_handle.config.get("game", "module_name"))

            # exit(0)

        # 判断是否使能了SD
        if self.sd_config["enable"]:
            from utils.sd import SD

            self.sd = SD(self.sd_config)

        # 判断是否使能了点歌模式
        if self.choose_song_config["enable"]:
            # 获取本地音频文件夹内所有的音频文件名
            self.choose_song_song_lists = My_handle.audio.get_dir_audios_filename(
                self.choose_song_config["song_path"])

        # 日志文件路径
        self.log_file_path = "./log/log-" + \
            My_handle.common.get_bj_time(1) + ".txt"
        if os.path.isfile(self.log_file_path):
            logging.info(f'{self.log_file_path} 日志文件已存在，跳过')
        else:
            with open(self.log_file_path, 'w') as f:
                f.write('')
                logging.info(f'{self.log_file_path} 日志文件已创建')

        self.comment_file_path = "./log/comment-" + \
            My_handle.common.get_bj_time(1) + ".txt"
        if os.path.isfile(self.comment_file_path):
            logging.info(f'{self.comment_file_path} 弹幕文件已存在，跳过')
        else:
            with open(self.comment_file_path, 'w') as f:
                f.write('')
                logging.info(f'{self.comment_file_path} 弹幕文件已创建')

        """

            .............  '>)xcn)I
            }}}}}}}}}}}}](v0kaaakad\..
            ++++++~~++<_xpahhhZ0phah>
            _________+(OhhkamuCbkkkh+
            ?????????nbhkhkn|makkkhQ^
            [[[[[[[}UhkbhZ]fbhkkkhb<
            1{1{1{1ChkkaXicohkkkhk]
            ))))))JhkkhrICakkkkap-
            \\\\|ckkkat;0akkkka0>
            ttt/fpkka/;Oakhhaku"
            jjjjUmkau^QabwQX\< '!<++~>iI       .;>++++<>I'     :+}}{?;
            xxxcpdkO"capmmZ/^ +Y-;,,;-Lf     ItX/+l:",;>1cx>  .`"x#d>`        .`.
            uuvqwkh+1ahaaL_  'Zq;     ;~   '/bQ!         "uhc: . 1oZ'         "vj.     ^'
            ccc0kaz!kawX}'   .\hbv?:      .jop;           .C*L^  )oO`        .':I^. ."_L!^^.    ':;,'
            XXXXph_cU_"        >rZhbC\!   "qaC...          faa~  )oO`        ;-jqj .l[mb1]_'  ^(|}\Ow{
            XXXz00i+             '!1Ukkc, 'JoZ` .          uop;  )oO'          >ou   .Lp"  . ,0j^^>Yvi
            XXXzLn. .        ^>      lC#(  lLot.          _kq- . 1o0'          >on   .Qp,    }*|><i^  .
            YYYXQ|           ,O]^.   "XQI . `10c~^.    '!t0f:   .t*q;....'l1. ._#c.. .Qkl`I_"Iw0~"`,<|i.
            (|((f1           ^t1]++-}(?`      '>}}}/rrx1]~^    ^?jvv/]--]{r) .i{x/+;  ]Xr1_;. :(vnrj\i.
                '1..             .''.   .         .Itq*Z}`             ..
                 +; .                                "}XmQf-i!;.
                  .                                     ';><iI"


        """
        try:
            # 数据库
            self.db = SQLiteDB(My_handle.config.get("database", "path"))
            logging.info(f'创建数据库:{My_handle.config.get("database", "path")}')

            # 创建弹幕表
            create_table_sql = '''
            CREATE TABLE IF NOT EXISTS danmu (
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                ts DATETIME NOT NULL
            )
            '''
            self.db.execute(create_table_sql)
            logging.info('创建danmu（弹幕）表')

            create_table_sql = '''
            CREATE TABLE IF NOT EXISTS entrance (
                username TEXT NOT NULL,
                ts DATETIME NOT NULL
            )
            '''
            self.db.execute(create_table_sql)
            logging.info('创建entrance（入场）表')

            create_table_sql = '''
            CREATE TABLE IF NOT EXISTS gift (
                username TEXT NOT NULL,
                gift_name TEXT NOT NULL,
                gift_num INT NOT NULL,
                unit_price REAL NOT NULL,
                total_price REAL NOT NULL,
                ts DATETIME NOT NULL
            )
            '''
            self.db.execute(create_table_sql)
            logging.info('创建gift（礼物）表')

            create_table_sql = '''
            CREATE TABLE IF NOT EXISTS integral (
                platform TEXT NOT NULL,
                username TEXT NOT NULL,
                uid TEXT NOT NULL,
                integral INT NOT NULL,
                view_num INT NOT NULL,
                sign_num INT NOT NULL,
                last_sign_ts DATETIME NOT NULL,
                total_price INT NOT NULL,
                last_ts DATETIME NOT NULL
            )
            '''
            self.db.execute(create_table_sql)
            logging.info('创建integral（积分）表')
        except Exception as e:
            logging.error(traceback.format_exc())

    def get_room_id(self):
        return self.room_id

    # 从本地问答库中搜索问题的答案

    def find_answer(self, question, qa_file_path, similarity=1):
        """从本地问答库中搜索问题的答案

        Args:
            question (str): 问题文本
            qa_file_path (str): 问答库的路径
            similarity (float): 相似度

        Returns:
            str: 答案文本 或 None
        """

        with open(qa_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        q_list = [lines[i].strip() for i in range(0, len(lines), 2)]
        q_to_answer_index = {q: i + 1 for i, q in enumerate(q_list)}

        q = My_handle.common.find_best_match(question, q_list, similarity)
        # print(f"q={q}")

        if q is not None:
            answer_index = q_to_answer_index.get(q)
            # print(f"answer_index={answer_index}")
            if answer_index is not None and answer_index < len(lines):
                return lines[answer_index * 2 - 1].strip()

        return None

    # 本地问答库 文本模式  根据相似度查找答案

    def find_similar_answer(self, input_str, qa_file_path, min_similarity=0.8):
        """本地问答库 文本模式  根据相似度查找答案

        Args:
            input_str (str): 输入的待查找字符串
            qa_file_path (str): 问答库的路径
            min_similarity (float, optional): 最低匹配相似度. 默认 0.8.

        Returns:
            response (str): 匹配到的结果，如果匹配不到则返回None
        """
        def load_data_from_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    return data
            except (FileNotFoundError, json.JSONDecodeError):
                return None

        # 从文件加载数据
        data = load_data_from_file(qa_file_path)
        if data is None:
            return None

        # 存储相似度与回答的元组列表
        similarity_responses = []

        # 遍历json中的每个条目，找到与输入字符串相似的关键词
        for entry in data:
            for keyword in entry.get("关键词", []):
                similarity = difflib.SequenceMatcher(
                    None, input_str, keyword).ratio()
                similarity_responses.append((similarity, entry.get("回答", [])))

        # 过滤相似度低于设定阈值的回答
        similarity_responses = [(similarity, response) for similarity,
                                response in similarity_responses if similarity >= min_similarity]

        # 如果没有符合条件的回答，返回None
        if not similarity_responses:
            return None

        # 按相似度降序排序
        similarity_responses.sort(reverse=True, key=lambda x: x[0])

        # 获取相似度最高的回答列表
        top_response = similarity_responses[0][1]

        # 随机选择一个回答
        response = random.choice(top_response)

        return response

    # 本地问答库 处理

    def local_qa_handle(self, data):
        """本地问答库 处理

        Args:
            data (dict): 用户名 弹幕数据

        Returns:
            bool: 是否触发并处理
        """
        user_name = data["username"]
        content = data["content"]

        # 合并字符串末尾连续的*  主要针对获取不到用户名的情况
        user_name = My_handle.common.merge_consecutive_asterisks(user_name)

        # 1、匹配本地问答库 触发后不执行后面的其他功能
        if self.local_qa["text"]["enable"] == True:
            # 根据类型，执行不同的问答匹配算法
            if self.local_qa["text"]["type"] == "text":
                tmp = self.find_answer(
                    content, self.local_qa["text"]["file_path"], self.local_qa["text"]["similarity"])
            else:
                tmp = self.find_similar_answer(
                    content, self.local_qa["text"]["file_path"], self.local_qa["text"]["similarity"])

            if tmp != None:
                logging.info(f"触发本地问答库-文本 [{user_name}]: {content}")
                # 将问答库中设定的参数替换为指定内容，开发者可以自定义替换内容
                if "{cur_time}" in tmp:
                    tmp = tmp.format(cur_time=My_handle.common.get_bj_time(5))
                if "{username}" in tmp:
                    tmp = tmp.format(username=user_name)
                else:
                    tmp = tmp

                logging.info(f"本地问答库-文本回答为: {tmp}")

                resp_content = tmp
                # 将 AI 回复记录到日志文件中
                with open(self.comment_file_path, "r+", encoding="utf-8") as f:
                    tmp_content = f.read()
                    # 将指针移到文件头部位置（此目的是为了让直播中读取日志文件时，可以一直让最新内容显示在顶部）
                    f.seek(0, 0)
                    # 不过这个实现方式，感觉有点低效
                    # 设置单行最大字符数，主要目的用于接入直播弹幕显示时，弹幕过长导致的显示溢出问题
                    max_length = 20
                    resp_content_substrings = [resp_content[i:i + max_length] for i in
                                               range(0, len(resp_content), max_length)]
                    resp_content_joined = '\n'.join(resp_content_substrings)

                    # 根据 弹幕日志类型进行各类日志写入
                    if My_handle.config.get("comment_log_type") == "问答":
                        f.write(
                            f"[{user_name} 提问]:{content}\n[AI回复{user_name}]:{resp_content_joined}\n" + tmp_content)
                    elif My_handle.config.get("comment_log_type") == "问题":
                        f.write(f"[{user_name} 提问]:{content}\n" + tmp_content)
                    elif My_handle.config.get("comment_log_type") == "回答":
                        f.write(
                            f"[AI回复{user_name}]:{resp_content_joined}\n" + tmp_content)

                message = {
                    "type": "comment",
                    "tts_type": My_handle.audio_synthesis_type,
                    "data": My_handle.config.get(My_handle.audio_synthesis_type),
                    "config": self.filter_config,
                    "user_name": user_name,
                    "content": resp_content
                }

                # 音频合成（edge-tts / vits_fast）并播放
                My_handle.audio.audio_synthesis(message)

                return True

        # 2、匹配本地问答音频库 触发后不执行后面的其他功能
        if self.local_qa["audio"]["enable"] == True:
            # 输出当前用户发送的弹幕消息
            # logging.info(f"[{user_name}]: {content}")
            # 获取本地问答音频库文件夹内所有的音频文件名
            local_qa_audio_filename_list = My_handle.audio.get_dir_audios_filename(
                self.local_qa["audio"]["file_path"], type=1)
            self.local_qa_audio_list = My_handle.audio.get_dir_audios_filename(
                self.local_qa["audio"]["file_path"], type=0)

            # 不含拓展名做查找
            local_qv_audio_filename = My_handle.common.find_best_match(
                content, local_qa_audio_filename_list, self.local_qa["audio"]["similarity"])

            # print(f"local_qv_audio_filename={local_qv_audio_filename}")

            # 找到了匹配的结果
            if local_qv_audio_filename is not None:
                logging.info(f"触发本地问答库-语音 [{user_name}]: {content}")
                # 把结果从原文件名列表中在查找一遍，补上拓展名
                local_qv_audio_filename = My_handle.common.find_best_match(
                    local_qv_audio_filename, self.local_qa_audio_list, 0)

                # 寻找对应的文件
                resp_content = My_handle.audio.search_files(
                    self.local_qa["audio"]["file_path"], local_qv_audio_filename)
                if resp_content != []:
                    logging.debug(f"匹配到的音频原相对路径：{resp_content[0]}")

                    # 拼接音频文件路径
                    resp_content = f'{self.local_qa["audio"]["file_path"]}/{resp_content[0]}'
                    logging.info(f"匹配到的音频路径：{resp_content}")
                    message = {
                        "type": "local_qa_audio",
                        "tts_type": My_handle.audio_synthesis_type,
                        "data": My_handle.config.get(My_handle.audio_synthesis_type),
                        "config": self.filter_config,
                        "user_name": user_name,
                        "content": content,
                        "file_path": resp_content
                    }

                    # 音频合成（edge-tts / vits_fast）并播放
                    My_handle.audio.audio_synthesis(message)

                    return True

        return False

    # 点歌模式 处理

    def choose_song_handle(self, data):
        """点歌模式 处理

        Args:
            data (dict): 用户名 弹幕数据

        Returns:
            bool: 是否触发并处理
        """
        user_name = data["username"]
        content = data["content"]

        # 合并字符串末尾连续的*  主要针对获取不到用户名的情况
        user_name = My_handle.common.merge_consecutive_asterisks(user_name)

        if self.choose_song_config["enable"] == True:
            # 判断点歌命令是否正确
            if content.startswith(self.choose_song_config["start_cmd"]):
                logging.info(f"[{user_name}]: {content}")

                # 去除命令前缀
                content = content[len(self.choose_song_config["start_cmd"]):]
                # 判断是否有此歌曲
                song_filename = My_handle.common.find_best_match(
                    content, self.choose_song_song_lists)
                if song_filename is None:
                    # resp_content = f"抱歉，我还没学会唱{content}"
                    # 根据配置的 匹配失败回复文案来进行合成
                    resp_content = self.choose_song_config["match_fail_copy"].format(
                        content=content)
                    logging.info(f"[AI回复{user_name}]：{resp_content}")

                    message = {
                        "type": "comment",
                        "tts_type": My_handle.audio_synthesis_type,
                        "data": My_handle.config.get(My_handle.audio_synthesis_type),
                        "config": self.filter_config,
                        "user_name": user_name,
                        "content": resp_content
                    }

                    # 音频合成（edge-tts / vits_fast）并播放
                    My_handle.audio.audio_synthesis(message)

                    return True

                resp_content = My_handle.audio.search_files(
                    self.choose_song_config['song_path'], song_filename)
                if resp_content == []:
                    return True

                logging.debug(f"匹配到的音频原相对路径：{resp_content[0]}")

                # 拼接音频文件路径
                resp_content = f"{self.choose_song_config['song_path']}/{resp_content[0]}"
                logging.info(f"匹配到的音频路径：{resp_content}")
                message = {
                    "type": "song",
                    "tts_type": My_handle.audio_synthesis_type,
                    "data": My_handle.config.get(My_handle.audio_synthesis_type),
                    "config": self.filter_config,
                    "user_name": user_name,
                    "content": resp_content
                }

                # 音频合成（edge-tts / vits_fast）并播放
                My_handle.audio.audio_synthesis(message)

                return True
            # 判断取消点歌命令是否正确
            elif content.startswith(self.choose_song_config["stop_cmd"]):
                My_handle.audio.stop_current_audio()

                return True
            # 判断随机点歌命令是否正确
            elif content == self.choose_song_config["random_cmd"]:
                resp_content = My_handle.common.random_search_a_audio_file(
                    self.choose_song_config['song_path'])
                if resp_content is None:
                    return True

                logging.info(f"随机到的音频路径：{resp_content}")

                message = {
                    "type": "song",
                    "tts_type": My_handle.audio_synthesis_type,
                    "data": My_handle.config.get(My_handle.audio_synthesis_type),
                    "config": self.filter_config,
                    "user_name": user_name,
                    "content": resp_content
                }

                # 音频合成（edge-tts / vits_fast）并播放
                My_handle.audio.audio_synthesis(message)

                return True

        return False

    # 画图模式 SD 处理

    def sd_handle(self, data):
        """画图模式 SD 处理

        Args:
            data (dict): 用户名 弹幕数据

        Returns:
            bool: 是否触发并处理
        """
        user_name = data["username"]
        content = data["content"]

        # 合并字符串末尾连续的*  主要针对获取不到用户名的情况
        user_name = My_handle.common.merge_consecutive_asterisks(user_name)

        if content.startswith(self.sd_config["trigger"]):
            # 含有违禁词/链接
            if My_handle.common.profanity_content(content) or My_handle.common.check_sensitive_words2(
                    self.filter_config["badwords_path"], content) or \
                    My_handle.common.is_url_check(content):
                logging.warning(f"违禁词/链接：{content}")
                return

            if self.sd_config["enable"] == False:
                logging.info("您还未启用SD模式，无法使用画画功能")
                return True
            else:
                # 输出当前用户发送的弹幕消息
                logging.info(f"[{user_name}]: {content}")

                content = content[len(self.sd_config["trigger"]):]

                # 根据设定的LLM
                if self.sd_config["prompt_llm"]["type"] == "chatgpt":
                    if self.chatgpt is None:
                        self.chatgpt = GPT_MODEL.get("chatgpt")

                    content = self.sd_config["prompt_llm"]["before_prompt"] + \
                        content + self.after_prompt
                    # 调用gpt接口，获取返回内容
                    resp_content = self.chatgpt.get_gpt_resp(
                        user_name, content)
                    if resp_content is not None:
                        # 输出 ChatGPT 返回的回复消息
                        logging.info(f"[AI回复{user_name}]：{resp_content}")
                    else:
                        resp_content = ""
                        logging.warning("警告：chatgpt无返回")
                elif self.sd_config["prompt_llm"]["type"] == "claude":
                    if self.claude is None:
                        self.claude = GPT_MODEL.get(self.chat_type)

                        # 初次运行 先重置下会话
                        if not self.claude.reset_claude():
                            logging.error("重置Claude会话失败喵~")

                    content = self.before_prompt + content + self.after_prompt
                    resp_content = self.claude.get_claude_resp(content)
                    if resp_content is not None:
                        # 输出 返回的回复消息
                        logging.info(f"[AI回复{user_name}]：{resp_content}")
                    else:
                        resp_content = ""
                        logging.warning("警告：claude无返回")
                elif self.sd_config["prompt_llm"]["type"] == "claude2":
                    if self.claude2 is None:
                        self.claude2 = GPT_MODEL.get(self.chat_type)

                        # 初次运行 先重置下会话
                        if self.claude2.get_organization_id() is None:
                            logging.error("重置Claude2会话失败喵~")

                    content = self.before_prompt + content + self.after_prompt
                    resp_content = self.claude2.get_claude2_resp(content)
                    if resp_content is not None:
                        # 输出 返回的回复消息
                        logging.info(f"[AI回复{user_name}]：{resp_content}")
                    else:
                        resp_content = ""
                        logging.warning("警告：claude2无返回")
                elif self.sd_config["prompt_llm"]["type"] == "chatglm":
                    if self.chatglm is None:
                        self.chatglm = GPT_MODEL.get(self.chat_type)

                    # 生成回复
                    resp_content = self.chatglm.get_chatglm_resp(content)
                    if resp_content is not None:
                        # 输出 返回的回复消息
                        logging.info(f"[AI回复{user_name}]：{resp_content}")
                    else:
                        resp_content = ""
                        logging.warning("警告：chatglm无返回")
                elif self.sd_config["prompt_llm"]["type"] == "text_generation_webui":
                    if self.text_generation_webui is None:
                        self.text_generation_webui = GPT_MODEL.get(
                            self.chat_type)

                    # 生成回复
                    resp_content = self.text_generation_webui.get_text_generation_webui_resp(
                        content)
                    if resp_content is not None:
                        # 输出 返回的回复消息
                        logging.info(f"[AI回复{user_name}]：{resp_content}")
                    else:
                        resp_content = ""
                        logging.warning("警告：text_generation_webui无返回")
                elif self.sd_config["prompt_llm"]["type"] == "none":
                    resp_content = content
                else:
                    resp_content = content

                self.sd.process_input(resp_content)
                return True

        return False

    # 弹幕格式检查和特殊字符替换

    def comment_check_and_replace(self, content):
        """弹幕格式检查和特殊字符替换

        Args:
            content (str): 待处理的弹幕内容

        Returns:
            str: 处理完毕后的弹幕内容/None
        """
        # 判断弹幕是否以xx起始，如果不是则返回
        if self.filter_config["before_must_str"] and not any(
                content.startswith(prefix) for prefix in self.filter_config["before_must_str"]):
            return None
        else:
            for prefix in self.filter_config["before_must_str"]:
                if content.startswith(prefix):
                    content = content[len(prefix):]  # 删除匹配的开头
                    break

        # 判断弹幕是否以xx结尾，如果不是则返回
        if self.filter_config["after_must_str"] and not any(
                content.endswith(prefix) for prefix in self.filter_config["after_must_str"]):
            return None
        else:
            for prefix in self.filter_config["after_must_str"]:
                if content.endswith(prefix):
                    content = content[:-len(prefix)]  # 删除匹配的结尾
                    break

        # 全为标点符号
        if My_handle.common.is_punctuation_string(content):
            return None

        # 换行转为,
        content = content.replace('\n', ',')

        # 语言检测
        if My_handle.common.lang_check(content, self.need_lang) is None:
            logging.warning("语言检测不通过，已过滤")
            return None

        return content

    # 违禁处理

    def prohibitions_handle(self, content):
        """违禁处理

        Args:
            content (str): 带判断的字符串内容

        Returns:
            bool: 是否违禁词 是True 否False
        """
        # 含有违禁词/链接
        if My_handle.common.profanity_content(content) or My_handle.common.is_url_check(content):
            logging.warning(f"违禁词/链接：{content}")
            return True

        # 违禁词过滤
        if self.filter_config["badwords_path"] != "":
            if My_handle.common.check_sensitive_words2(self.filter_config["badwords_path"], content):
                logging.warning(f"本地违禁词：{content}")
                return True

        # 同拼音违禁词过滤
        if self.filter_config["bad_pinyin_path"] != "":
            if My_handle.common.check_sensitive_words3(self.filter_config["bad_pinyin_path"], content):
                logging.warning(f"同音违禁词：{content}")
                return True

        return False

    # 直接复读

    def reread_handle(self, data):
        """复读处理

        Args:
            data (dict): 包含用户名,弹幕内容

        Returns:
            _type_: 寂寞
        """

        user_name = data["user_name"]
        content = data["content"]

        logging.info(f"复读内容：{content}")

        # 音频合成时需要用到的重要数据
        message = {
            "type": "reread",
            "tts_type": My_handle.audio_synthesis_type,
            "data": My_handle.config.get(My_handle.audio_synthesis_type),
            "config": self.filter_config,
            "user_name": user_name,
            "content": content
        }

        My_handle.audio.audio_synthesis(message)

    # LLM处理

    def llm_handle(self, chat_type, data):
        """LLM统一处理

        Args:
            chat_type (str): 聊天类型
            data (str): dict，含用户名和内容

        Returns:
            str: LLM返回的结果
        """
        resp_content = None

        if chat_type == "chatgpt":
            # 调用gpt接口，获取返回内容
            resp_content = self.chatgpt.get_gpt_resp(
                data["user_name"], data["content"])
        elif chat_type == "claude":
            resp_content = self.claude.get_claude_resp(data["content"])
        elif chat_type == "claude2":
            resp_content = self.claude2.get_claude2_resp(data["content"])
        elif chat_type == "chatterbot":
            # 生成回复
            resp_content = self.bot.get_response(data["content"]).text
        elif chat_type == "chatglm":
            resp_content = self.chatglm.get_chatglm_resp(data["content"])
        elif chat_type == "chat_with_file":
            resp_content = self.chat_with_file.get_model_resp(data["content"])
        elif chat_type == "text_generation_webui":
            # 生成回复
            resp_content = self.text_generation_webui.get_text_generation_webui_resp(
                data["content"])
        elif chat_type == "sparkdesk":
            # 生成回复
            resp_content = self.sparkdesk.get_sparkdesk_resp(data["content"])
        elif chat_type == "langchain_chatglm":
            # 生成回复
            resp_content = self.langchain_chatglm.get_resp(data["content"])
        elif chat_type == "zhipu":
            # 生成回复
            resp_content = self.zhipu.get_resp(data["content"])
        elif chat_type == "bard":
            # 生成回复
            resp_content = self.bard_api.get_resp(data["content"])
        elif chat_type == "yiyan":
            # 生成回复
            resp_content = self.yiyan.get_resp(data["content"])
        elif chat_type == "tongyi":
            # 生成回复
            resp_content = self.tongyi.get_resp(data["content"])
        elif chat_type == "reread":
            # 复读机
            resp_content = data["content"]
        elif chat_type == "none":
            # 不启用
            pass
        else:
            resp_content = data["content"]

        return resp_content

    # 积分处理

    def integral_handle(self, type, data):
        """积分处理

        Args:
            type (str): 消息数据类型（comment/gift/entrance）
            data (dict): 平台侧传入的data数据，直接拿来做解析

        Returns:
            bool: 是否正常触发了积分事件，是True 否False
        """
        user_name = data["username"]

        if My_handle.config.get("integral", "enable"):
            # 根据消息类型进行对应处理
            if "comment" == type:
                content = data["content"]

                # 是否开启了签到功能
                if My_handle.config.get("integral", "sign", "enable"):
                    # 判断弹幕内容是否是命令
                    if content in My_handle.config.get("integral", "sign", "cmd"):
                        # 查询数据库中是否有当前用户的积分记录（缺个UID）
                        common_sql = '''
                        SELECT * FROM integral WHERE username =?
                        '''
                        integral_data = self.db.fetch_all(
                            common_sql, (user_name,))

                        logging.debug(f"integral_data={integral_data}")

                        # 获取文案并合成语音，传入签到天数自动检索
                        def get_copywriting_and_audio_synthesis(sign_num):
                            # 判断当前签到天数在哪个签到数区间内，根据不同的区间提供不同的文案回复
                            for integral_sign_copywriting in My_handle.config.get("integral", "sign", "copywriting"):
                                # 在此区间范围内，所以你的配置一定要对，不然这里就崩溃了！！！
                                if int(integral_sign_copywriting["sign_num_interval"].split("-")[0]) <= \
                                        sign_num <= \
                                        int(integral_sign_copywriting["sign_num_interval"].split("-")[1]):
                                    # 匹配文案
                                    resp_content = random.choice(
                                        integral_sign_copywriting["copywriting"])

                                    logging.debug(
                                        f"resp_content={resp_content}")

                                    data_json = {
                                        "user_name": data["username"],
                                        "get_integral": int(My_handle.config.get("integral", "sign", "get_integral")),
                                        "sign_num": sign_num + 1
                                    }

                                    resp_content = self.common.dynamic_variable_replacement(
                                        resp_content, data_json)

                                    # 生成回复内容
                                    message = {
                                        "type": "direct_reply",
                                        "tts_type": My_handle.audio_synthesis_type,
                                        "data": My_handle.config.get(My_handle.audio_synthesis_type),
                                        "config": self.filter_config,
                                        "user_name": user_name,
                                        "content": resp_content
                                    }

                                    # 音频合成（edge-tts / vits_fast）并播放
                                    My_handle.audio.audio_synthesis(message)

                        if integral_data == []:
                            # 积分表中没有该用户，插入数据
                            insert_data_sql = '''
                            INSERT INTO integral (platform, username, uid, integral, view_num, sign_num, last_sign_ts, total_price, last_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            '''
                            self.db.execute(insert_data_sql, (
                                data["platform"],
                                user_name,
                                user_name,
                                My_handle.config.get(
                                    "integral", "sign", "get_integral"),
                                1,
                                1,
                                datetime.now(),
                                0,
                                datetime.now())
                            )

                            logging.info(f"integral积分表 新增 用户：{user_name}")

                            get_copywriting_and_audio_synthesis(0)

                            return True
                        else:
                            integral_data = integral_data[0]
                            # 积分表中有该用户，更新数据

                            # 先判断last_sign_ts是否是今天，如果是，则说明已经打卡过了，不能重复打卡
                            # 获取日期时间字符串字段，此处是个坑点，一旦数据库结构发生改变或者select语句改了，就会关联影响！！！
                            date_string = integral_data[6]

                            # 获取日期部分（前10个字符），并与当前日期字符串比较
                            if date_string[:10] == datetime.now().date().strftime("%Y-%m-%d"):
                                message = {
                                    "type": "direct_reply",
                                    "tts_type": My_handle.audio_synthesis_type,
                                    "data": My_handle.config.get(My_handle.audio_synthesis_type),
                                    "config": self.filter_config,
                                    "user_name": user_name,
                                    "content": f"{user_name}您今天已经签到过了，不能重复打卡哦~"
                                }

                                # 音频合成（edge-tts / vits_fast）并播放
                                My_handle.audio.audio_synthesis(message)

                                return True

                            # 更新下用户数据
                            update_data_sql = '''
                            UPDATE integral SET integral=?, view_num=?, sign_num=?, last_sign_ts=?, last_ts=? WHERE username =?
                            '''
                            self.db.execute(update_data_sql, (
                                # 此处是个坑点，一旦数据库结构发生改变或者select语句改了，就会关联影响！！！
                                integral_data[3] + My_handle.config.get(
                                    "integral", "sign", "get_integral"),
                                integral_data[4] + 1,
                                integral_data[5] + 1,
                                datetime.now(),
                                datetime.now(),
                                user_name
                            )
                            )

                            logging.info(f"integral积分表 更新 用户：{user_name}")

                            get_copywriting_and_audio_synthesis(
                                integral_data[5] + 1)

                            return True
            elif "gift" == type:
                # 是否开启了礼物功能
                if My_handle.config.get("integral", "gift", "enable"):
                    # 查询数据库中是否有当前用户的积分记录（缺个UID）
                    common_sql = '''
                    SELECT * FROM integral WHERE username =?
                    '''
                    integral_data = self.db.fetch_all(common_sql, (user_name,))

                    logging.debug(f"integral_data={integral_data}")

                    get_integral = int(float(My_handle.config.get(
                        "integral", "gift", "get_integral_proportion")) * data["total_price"])

                    # 获取文案并合成语音，传入总礼物金额自动检索
                    def get_copywriting_and_audio_synthesis(total_price):
                        # 判断当前礼物金额在哪个礼物金额区间内，根据不同的区间提供不同的文案回复
                        for integral_gift_copywriting in My_handle.config.get("integral", "gift", "copywriting"):
                            # 在此区间范围内，所以你的配置一定要对，不然这里就崩溃了！！！
                            if float(integral_gift_copywriting["gift_price_interval"].split("-")[0]) <= \
                                    total_price <= \
                                    float(integral_gift_copywriting["gift_price_interval"].split("-")[1]):
                                # 匹配文案
                                resp_content = random.choice(
                                    integral_gift_copywriting["copywriting"])

                                logging.debug(f"resp_content={resp_content}")

                                data_json = {
                                    "user_name": data["username"],
                                    "gift_name": data["gift_name"],
                                    "get_integral": get_integral
                                }

                                resp_content = self.common.dynamic_variable_replacement(
                                    resp_content, data_json)

                                # 生成回复内容
                                message = {
                                    "type": "direct_reply",
                                    "tts_type": My_handle.audio_synthesis_type,
                                    "data": My_handle.config.get(My_handle.audio_synthesis_type),
                                    "config": self.filter_config,
                                    "user_name": user_name,
                                    "content": resp_content
                                }

                                # 音频合成（edge-tts / vits_fast）并播放
                                My_handle.audio.audio_synthesis(message)

                    # TODO：此处有计算bug！！！ 总礼物价值计算不对，后期待优化
                    if integral_data == []:
                        # 积分表中没有该用户，插入数据
                        insert_data_sql = '''
                        INSERT INTO integral (platform, username, uid, integral, view_num, sign_num, last_sign_ts, total_price, last_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        '''
                        self.db.execute(insert_data_sql, (
                            data["platform"],
                            user_name,
                            user_name,
                            get_integral,
                            1,
                            1,
                            datetime.now(),
                            data["total_price"],
                            datetime.now())
                        )

                        logging.info(f"integral积分表 新增 用户：{user_name}")

                        get_copywriting_and_audio_synthesis(
                            data["total_price"])

                        return True
                    else:
                        integral_data = integral_data[0]
                        # 积分表中有该用户，更新数据

                        # 更新下用户数据
                        update_data_sql = '''
                        UPDATE integral SET integral=?, total_price=?, last_ts=? WHERE username =?
                        '''
                        self.db.execute(update_data_sql, (
                            # 此处是个坑点，一旦数据库结构发生改变或者select语句改了，就会关联影响！！！
                            integral_data[3] + get_integral,
                            integral_data[7] + data["total_price"],
                            datetime.now(),
                            user_name
                        )
                        )

                        logging.info(f"integral积分表 更新 用户：{user_name}")

                        get_copywriting_and_audio_synthesis(
                            data["total_price"])

                        return True
            elif "entrance" == type:
                # 是否开启了入场功能
                if My_handle.config.get("integral", "entrance", "enable"):
                    # 查询数据库中是否有当前用户的积分记录（缺个UID）
                    common_sql = '''
                    SELECT * FROM integral WHERE username =?
                    '''
                    integral_data = self.db.fetch_all(common_sql, (user_name,))

                    logging.debug(f"integral_data={integral_data}")

                    # 获取文案并合成语音，传入观看天数自动检索
                    def get_copywriting_and_audio_synthesis(view_num):
                        # 判断当前签到天数在哪个签到数区间内，根据不同的区间提供不同的文案回复
                        for integral_entrance_copywriting in My_handle.config.get("integral", "entrance", "copywriting"):
                            # 在此区间范围内，所以你的配置一定要对，不然这里就崩溃了！！！
                            if int(integral_entrance_copywriting["entrance_num_interval"].split("-")[0]) <= \
                                    view_num <= \
                                    int(integral_entrance_copywriting["entrance_num_interval"].split("-")[1]):
                                # 匹配文案
                                resp_content = random.choice(
                                    integral_entrance_copywriting["copywriting"])

                                logging.debug(f"resp_content={resp_content}")

                                data_json = {
                                    "user_name": data["username"],
                                    "get_integral": int(My_handle.config.get("integral", "entrance", "get_integral")),
                                    "entrance_num": view_num + 1
                                }

                                resp_content = self.common.dynamic_variable_replacement(
                                    resp_content, data_json)

                                # 生成回复内容
                                message = {
                                    "type": "direct_reply",
                                    "tts_type": My_handle.audio_synthesis_type,
                                    "data": My_handle.config.get(My_handle.audio_synthesis_type),
                                    "config": self.filter_config,
                                    "user_name": user_name,
                                    "content": resp_content
                                }

                                # 音频合成（edge-tts / vits_fast）并播放
                                My_handle.audio.audio_synthesis(message)

                    if integral_data == []:
                        # 积分表中没有该用户，插入数据
                        insert_data_sql = '''
                        INSERT INTO integral (platform, username, uid, integral, view_num, sign_num, last_sign_ts, total_price, last_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        '''
                        self.db.execute(insert_data_sql, (
                            data["platform"],
                            user_name,
                            user_name,
                            My_handle.config.get(
                                "integral", "entrance", "get_integral"),
                            1,
                            0,
                            datetime.now(),
                            0,
                            datetime.now())
                        )

                        logging.info(f"integral积分表 新增 用户：{user_name}")

                        get_copywriting_and_audio_synthesis(1)

                        return True
                    else:
                        integral_data = integral_data[0]
                        # 积分表中有该用户，更新数据

                        # 先判断last_ts是否是今天，如果是，则说明已经观看过了，不能重复记录
                        # 获取日期时间字符串字段，此处是个坑点，一旦数据库结构发生改变或者select语句改了，就会关联影响！！！
                        date_string = integral_data[8]

                        # 获取日期部分（前10个字符），并与当前日期字符串比较
                        if date_string[:10] == datetime.now().date().strftime("%Y-%m-%d"):
                            return False

                        # 更新下用户数据
                        update_data_sql = '''
                        UPDATE integral SET integral=?, view_num=?, last_ts=? WHERE username =?
                        '''
                        self.db.execute(update_data_sql, (
                            # 此处是个坑点，一旦数据库结构发生改变或者select语句改了，就会关联影响！！！
                            integral_data[3] + My_handle.config.get(
                                "integral", "entrance", "get_integral"),
                            integral_data[4] + 1,
                            datetime.now(),
                            user_name
                        )
                        )

                        logging.info(f"integral积分表 更新 用户：{user_name}")

                        get_copywriting_and_audio_synthesis(
                            integral_data[4] + 1)

                        return True
            elif "crud" == type:
                content = data["content"]

                # 是否开启了查询功能
                if My_handle.config.get("integral", "crud", "query", "enable"):
                    # 判断弹幕内容是否是命令
                    if content in My_handle.config.get("integral", "crud", "query", "cmd"):
                        # 查询数据库中是否有当前用户的积分记录（缺个UID）
                        common_sql = '''
                        SELECT * FROM integral WHERE username =?
                        '''
                        integral_data = self.db.fetch_all(
                            common_sql, (user_name,))

                        logging.debug(f"integral_data={integral_data}")

                        # 获取文案并合成语音，传入积分总数自动检索
                        def get_copywriting_and_audio_synthesis(total_integral):
                            # 匹配文案
                            resp_content = random.choice(My_handle.config.get(
                                "integral", "crud", "query", "copywriting"))

                            logging.debug(f"resp_content={resp_content}")

                            data_json = {
                                "user_name": data["username"],
                                "integral": total_integral
                            }

                            resp_content = self.common.dynamic_variable_replacement(
                                resp_content, data_json)

                            # 如果积分为0，则返回个没积分的回复。不过这个基本没可能，除非有bug
                            if total_integral == 0:
                                resp_content = data["username"] + "，查询到您无积分。"

                            # 生成回复内容
                            message = {
                                "type": "direct_reply",
                                "tts_type": My_handle.audio_synthesis_type,
                                "data": My_handle.config.get(My_handle.audio_synthesis_type),
                                "config": self.filter_config,
                                "user_name": user_name,
                                "content": resp_content
                            }

                            # 音频合成（edge-tts / vits_fast）并播放
                            My_handle.audio.audio_synthesis(message)

                        if integral_data == []:
                            logging.info(f"integral积分表 查询不到 用户：{user_name}")

                            get_copywriting_and_audio_synthesis(0)

                            return True
                        else:
                            integral_data = integral_data[0]
                            # 积分表中有该用户

                            # 获取日期时间字符串字段，此处是个坑点，一旦数据库结构发生改变或者select语句改了，就会关联影响！！！
                            date_string = integral_data[3]

                            logging.info(
                                f"integral积分表 用户：{user_name}，总积分：{date_string}")

                            get_copywriting_and_audio_synthesis(
                                int(date_string))

                            return True
        return False

    # 按键映射处理

    def key_mapping_handle(self, data):
        """按键映射处理

        Args:
            data (dict): 平台侧传入的data数据，直接拿来做解析

        Returns:
            bool: 是否正常触发了按键映射事件，是True 否False
        """
        # 官方文档：https://pyautogui.readthedocs.io/en/latest/keyboard.html#keyboard-keys
        if My_handle.config.get("key_mapping", "enable"):
            content = data["content"]
            # 判断命令头是否匹配
            start_cmd = My_handle.config.get("key_mapping", "start_cmd")
            if start_cmd != "" and content.startswith(start_cmd):
                # 删除命令头部
                content = content[len(start_cmd):]

            key_mapping_configs = My_handle.config.get("key_mapping", "config")

            for key_mapping_config in key_mapping_configs:
                similarity = float(key_mapping_config["similarity"])
                for keyword in key_mapping_config["keywords"]:
                    # 判断相似度
                    if keyword in content:
                        ratio = 1
                    else:
                        ratio = difflib.SequenceMatcher(
                            None, content, keyword).ratio()
                    if ratio >= similarity:
                        # 触发对应按键按下释放
                        for key in key_mapping_config["keys"]:
                            pyautogui.keyDown(key)
                        for key in key_mapping_config["keys"]:
                            pyautogui.keyUp(key)

                        logging.info(
                            f'【触发按键映射】关键词：{keyword} 按键：{key_mapping_config["keys"]}')

                        return True

        return False

    # 弹幕处理

    def comment_handle(self, data):
        """弹幕处理

        Args:
            data (dict): 包含用户名,弹幕内容

        Returns:
            _type_: 寂寞
        """

        try:
            user_name = data["username"]
            content = data["content"]

            # 记录数据库
            if My_handle.config.get("database", "comment_enable"):
                insert_data_sql = '''
                INSERT INTO danmu (username, content, ts) VALUES (?, ?, ?)
                '''
                self.db.execute(insert_data_sql,
                                (user_name, content, datetime.now()))

            # 合并字符串末尾连续的*  主要针对获取不到用户名的情况
            user_name = My_handle.common.merge_consecutive_asterisks(user_name)

            # 0、积分机制运转
            if self.integral_handle("comment", data):
                return
            if self.integral_handle("crud", data):
                return

            # 输出当前用户发送的弹幕消息
            logging.info(f"[{user_name}]: {content}")

            """
            用户名也得过滤一下，防止炸弹人
            """
            # 用户名以及弹幕违禁判断
            if self.prohibitions_handle(user_name) or self.prohibitions_handle(content):
                return

            # 弹幕格式检查和特殊字符替换
            content = self.comment_check_and_replace(content)
            if content is None:
                return

            # 判断字符串是否全为标点符号，是的话就过滤
            if My_handle.common.is_punctuation_string(content):
                logging.debug(f"用户:{user_name}]，发送纯符号的弹幕，已过滤")
                return

            # 按键映射 触发后不执行后面的其他功能
            if self.key_mapping_handle(data):
                return

            try:
                # 念弹幕
                if My_handle.config.get("read_comment", "enable"):
                    logging.debug(f"念弹幕 content:{content}")

                    # 音频合成时需要用到的重要数据
                    message = {
                        "type": "read_comment",
                        "tts_type": My_handle.audio_synthesis_type,
                        "data": My_handle.config.get(My_handle.audio_synthesis_type),
                        "config": self.filter_config,
                        "user_name": user_name,
                        "content": content
                    }

                    # 判断是否需要念用户名
                    if My_handle.config.get("read_comment", "read_username_enable"):
                        # 将用户名中特殊字符替换为空
                        message['user_name'] = self.common.replace_special_characters(
                            message['user_name'], "！!@#￥$%^&*_-+/——=()（）【】}|{:;<>~`\\")
                        tmp_content = random.choice(self.config.get(
                            "read_comment", "read_username_copywriting"))
                        if "{username}" in tmp_content:
                            message['content'] = tmp_content.format(
                                username=message['user_name']) + message['content']

                    # 音频合成（edge-tts / vits_fast）并播放
                    My_handle.audio.audio_synthesis(message)
            except Exception as e:
                logging.error(traceback.format_exc())

            # 1、本地问答库 处理
            if self.local_qa_handle(data):
                return

            # 2、点歌模式 触发后不执行后面的其他功能
            if self.choose_song_handle(data):
                return

            # 3、画图模式 触发后不执行后面的其他功能
            if self.sd_handle(data):
                return

            data_json = {
                "user_name": user_name,
                "content": content
            }

            """
            根据聊天类型执行不同逻辑
            """
            if self.chat_type == "chatgpt":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 ChatGPT 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning("警告：chatgpt无返回")
            elif self.chat_type == "claude":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning("警告：claude无返回")
            elif self.chat_type == "claude2":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning("警告：claude2无返回")
            elif self.chat_type == "chatterbot":
                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                logging.info(f"[AI回复{user_name}]：{resp_content}")
            elif self.chat_type == "chatglm":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning("警告：chatglm无返回")
            elif self.chat_type == "chat_with_file":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                print(f"[AI回复{user_name}]：{resp_content}")
            elif self.chat_type == "text_generation_webui":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning("警告：text_generation_webui无返回")
            elif self.chat_type == "sparkdesk":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning("警告：讯飞星火无返回")
            elif self.chat_type == "langchain_chatglm":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning("警告：langchain_chatglm无返回")
            elif self.chat_type == "zhipu":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning("警告：智谱AI无返回")
            elif self.chat_type == "bard":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning(
                        "警告：Bard无返回，请检查配置、网络是否正确，也可能是token过期，需要清空cookie重新登录获取")
            elif self.chat_type == "yiyan":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning(
                        "警告：文心一言无返回，请检查配置、网络是否正确，也可能是cookie过期或失效，需要重新获取cookie")
            elif self.chat_type == "tongyi":
                data_json["content"] = self.before_prompt + \
                    content + self.after_prompt

                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
                if resp_content is not None:
                    # 输出 返回的回复消息
                    logging.info(f"[AI回复{user_name}]：{resp_content}")
                else:
                    resp_content = ""
                    logging.warning(
                        "警告：通义千问无返回，请检查配置、网络是否正确，也可能是cookie过期或失效，需要重新获取cookie")
            elif self.chat_type == "game":
                # return

                if My_handle.config.get("game", "enable"):
                    # 传入切分后的弹幕内容
                    self.game.parse_keys_and_simulate_keys_press(
                        content.split(), 2)

                return
            elif self.chat_type == "reread":
                # 调用LLM统一接口，获取返回内容
                resp_content = self.llm_handle(self.chat_type, data_json)
            elif self.chat_type == "none":
                # 不启用
                return
            else:
                resp_content = content

            # 空数据结束
            if resp_content == "" or resp_content is None:
                return

            """
            双重过滤，为您保驾护航
            """
            resp_content = resp_content.replace('\n', '。')

            # LLM回复的内容进行违禁判断
            if self.prohibitions_handle(resp_content):
                return

            # logger.info("resp_content=" + resp_content)

            # 将 AI 回复记录到日志文件中
            with open(self.comment_file_path, "r+", encoding="utf-8") as f:
                tmp_content = f.read()
                # 将指针移到文件头部位置（此目的是为了让直播中读取日志文件时，可以一直让最新内容显示在顶部）
                f.seek(0, 0)
                # 不过这个实现方式，感觉有点低效
                # 设置单行最大字符数，主要目的用于接入直播弹幕显示时，弹幕过长导致的显示溢出问题
                max_length = 20
                resp_content_substrings = [resp_content[i:i + max_length]
                                           for i in range(0, len(resp_content), max_length)]
                resp_content_joined = '\n'.join(resp_content_substrings)

                # 根据 弹幕日志类型进行各类日志写入
                if My_handle.config.get("comment_log_type") == "问答":
                    f.write(
                        f"[{user_name} 提问]:\n{content}\n[AI回复{user_name}]:{resp_content_joined}\n" + tmp_content)
                elif My_handle.config.get("comment_log_type") == "问题":
                    f.write(f"[{user_name} 提问]:\n{content}\n" + tmp_content)
                elif My_handle.config.get("comment_log_type") == "回答":
                    f.write(
                        f"[AI回复{user_name}]:\n{resp_content_joined}\n" + tmp_content)

            # 音频合成时需要用到的重要数据
            message = {
                "type": "comment",
                "tts_type": My_handle.audio_synthesis_type,
                "data": My_handle.config.get(My_handle.audio_synthesis_type),
                "config": self.filter_config,
                "user_name": user_name,
                "content": resp_content
            }

            # 音频合成（edge-tts / vits_fast）并播放
            My_handle.audio.audio_synthesis(message)
        except Exception as e:
            logging.error(traceback.format_exc())

    # 礼物处理

    def gift_handle(self, data):
        try:
            # 记录数据库
            if My_handle.config.get("database", "gift_enable"):
                insert_data_sql = '''
                INSERT INTO gift (username, gift_name, gift_num, unit_price, total_price, ts) VALUES (?, ?, ?, ?, ?, ?)
                '''
                self.db.execute(insert_data_sql, (
                    data['username'],
                    data['gift_name'],
                    data['num'],
                    data['unit_price'],
                    data['total_price'],
                    datetime.now())
                )

            # 违禁处理
            if self.prohibitions_handle(data['username']):
                return

            if self.integral_handle("gift", data):
                return

            # 合并字符串末尾连续的*  主要针对获取不到用户名的情况
            data['username'] = My_handle.common.merge_consecutive_asterisks(
                data['username'])
            # 删除用户名中的特殊字符
            data['username'] = My_handle.common.replace_special_characters(
                data['username'], "！!@#￥$%^&*_-+/——=()（）【】}|{:;<>~`\\")

            # logging.debug(f"[{data['username']}]: {data}")

            if False == self.thanks_config["gift_enable"]:
                return

            # 如果礼物总价低于设置的礼物感谢最低值
            if data["total_price"] < self.thanks_config["lowest_price"]:
                return

            resp_content = self.thanks_config["gift_copy"].format(
                username=data["username"], gift_name=data["gift_name"])

            message = {
                "type": "gift",
                "tts_type": My_handle.audio_synthesis_type,
                "data": My_handle.config.get(My_handle.audio_synthesis_type),
                "config": self.filter_config,
                "user_name": data["username"],
                "content": resp_content
            }

            # 音频合成（edge-tts / vits_fast）并播放
            My_handle.audio.audio_synthesis(message)
        except Exception as e:
            logging.error(traceback.format_exc())

    # 入场处理

    def entrance_handle(self, data):
        try:
            # 记录数据库
            if My_handle.config.get("database", "entrance_enable"):
                insert_data_sql = '''
                INSERT INTO entrance (username, ts) VALUES (?, ?)
                '''
                self.db.execute(insert_data_sql,
                                (data['username'], datetime.now()))

            # 违禁处理
            if self.prohibitions_handle(data['username']):
                return

            if self.integral_handle("entrance", data):
                return

            # 合并字符串末尾连续的*  主要针对获取不到用户名的情况
            data['username'] = My_handle.common.merge_consecutive_asterisks(
                data['username'])
            # 删除用户名中的特殊字符
            data['username'] = My_handle.common.replace_special_characters(
                data['username'], "！!@#￥$%^&*_-+/——=()（）【】}|{:;<>~`\\")

            # logging.debug(f"[{data['username']}]: {data['content']}")

            if False == self.thanks_config["entrance_enable"]:
                return

            resp_content = self.thanks_config["entrance_copy"].format(
                username=data["username"])

            message = {
                "type": "entrance",
                "tts_type": My_handle.audio_synthesis_type,
                "data": My_handle.config.get(My_handle.audio_synthesis_type),
                "config": self.filter_config,
                "user_name": data['username'],
                "content": resp_content
            }

            # 音频合成（edge-tts / vits_fast）并播放
            My_handle.audio.audio_synthesis(message)
        except Exception as e:
            logging.error(traceback.format_exc())

    # 关注处理

    def follow_handle(self, data):
        try:
            # 合并字符串末尾连续的*  主要针对获取不到用户名的情况
            data['username'] = My_handle.common.merge_consecutive_asterisks(
                data['username'])
            # 删除用户名中的特殊字符
            data['username'] = My_handle.common.replace_special_characters(
                data['username'], "！!@#￥$%^&*_-+/——=()（）【】}|{:;<>~`\\")

            # 违禁处理
            if self.prohibitions_handle(data['username']):
                return

            # logging.debug(f"[{data['username']}]: {data['content']}")

            if False == self.thanks_config["follow_enable"]:
                return

            resp_content = self.thanks_config["follow_copy"].format(
                username=data["username"])

            message = {
                "type": "follow",
                "tts_type": My_handle.audio_synthesis_type,
                "data": My_handle.config.get(My_handle.audio_synthesis_type),
                "config": self.filter_config,
                "user_name": data['username'],
                "content": resp_content
            }

            # 音频合成（edge-tts / vits_fast）并播放
            My_handle.audio.audio_synthesis(message)
        except Exception as e:
            logging.error(traceback.format_exc())

    # 定时处理
    def schedule_handle(self, data):
        try:
            content = data["content"]

            message = {
                "type": "entrance",
                "tts_type": My_handle.audio_synthesis_type,
                "data": My_handle.config.get(My_handle.audio_synthesis_type),
                "config": self.filter_config,
                "user_name": data['username'],
                "content": content
            }

            # 音频合成（edge-tts / vits_fast）并播放
            My_handle.audio.audio_synthesis(message)
        except Exception as e:
            logging.error(traceback.format_exc())

    """
    数据丢弃部分
    增加新的处理事件时，需要进行这块部分的内容追加
    """

    def process_data(self, data, timer_flag):
        with self.data_lock:
            if timer_flag not in self.timers or not self.timers[timer_flag].is_alive():
                self.timers[timer_flag] = threading.Timer(self.get_interval(
                    timer_flag), self.process_last_data, args=(timer_flag,))
                self.timers[timer_flag].start()

            # self.timers[timer_flag].last_data = data
            if hasattr(self.timers[timer_flag], 'last_data'):
                self.timers[timer_flag].last_data.append(data)
                # 这里需要注意配置命名!!!
                if len(self.timers[timer_flag].last_data) > int(My_handle.config.get("filter", timer_flag + "_forget_reserve_num")):
                    self.timers[timer_flag].last_data.pop(0)
            else:
                self.timers[timer_flag].last_data = [data]

    def process_last_data(self, timer_flag):
        with self.data_lock:
            timer = self.timers.get(timer_flag)
            if timer and timer.last_data is not None and timer.last_data != []:
                logging.debug(
                    f"预处理定时器触发 type={timer_flag}，data={timer.last_data}")

                if timer_flag == "comment":
                    for data in timer.last_data:
                        self.comment_handle(data)
                elif timer_flag == "gift":
                    for data in timer.last_data:
                        self.gift_handle(data)
                    # self.gift_handle(timer.last_data)
                elif timer_flag == "entrance":
                    for data in timer.last_data:
                        self.entrance_handle(data)
                    # self.entrance_handle(timer.last_data)
                elif timer_flag == "follow":
                    for data in timer.last_data:
                        self.follow_handle(data)
                elif timer_flag == "talk":
                    # 聊天暂时共用弹幕处理逻辑
                    for data in timer.last_data:
                        self.comment_handle(data)
                    # self.comment_handle(timer.last_data)
                elif timer_flag == "schedule":
                    # 定时任务处理
                    for data in timer.last_data:
                        self.schedule_handle(data)
                    # self.schedule_handle(timer.last_data)

                # 清空数据
                timer.last_data = []

    def get_interval(self, timer_flag):
        # 根据标志定义不同计时器的间隔
        intervals = {
            "comment": My_handle.config.get("filter", "comment_forget_duration"),
            "gift": My_handle.config.get("filter", "gift_forget_duration"),
            "entrance": My_handle.config.get("filter", "entrance_forget_duration"),
            "follow": My_handle.config.get("filter", "follow_forget_duration"),
            "talk": My_handle.config.get("filter", "talk_forget_duration"),
            "schedule": My_handle.config.get("filter", "schedule_forget_duration")
            # 根据需要添加更多计时器及其间隔，记得添加config.json中的配置项
        }

        # 默认间隔为0.1秒
        return intervals.get(timer_flag, 0.1)
