"""Search normalisation: lower-case, and fold common Traditional Chinese characters to Simplified
(Whisper mixes both).  A compact table of frequent characters, not a full converter."""

_T = ("們這個說來時會對還麼國學點後發樣頭見實為開關長問間東車門馬鳥魚機電話語讀寫聽覺現應該給讓認識興歡樂愛親"
      "錢買賣貴飯飲館場園圖書筆紙練習題號碼網線遊戲錄視攝體動參與從邊過進運還這裡裏麼嗎嗬啊吧呢喔噢嘿"
      "們幾隻條張處觀覽廳廁層樓梯燈燒熱涼雲氣風颱雨雪節慶週週歲數萬億夠舊陽陰晚島灣海港澳臺韓與狀況總經濟")
_S = ("们这个说来时会对还么国学点后发样头见实为开关长问间东车门马鸟鱼机电话语读写听觉现应该给让认识兴欢乐爱亲"
      "钱买卖贵饭饮馆场园图书笔纸练习题号码网线游戏录视摄体动参与从边过进运还这里里么吗嗬啊吧呢喔噢嘿"
      "们几只条张处观览厅厕层楼梯灯烧热凉云气风台雨雪节庆周周岁数万亿够旧阳阴晚岛湾海港澳台韩与状况总经济")
assert len(_T) == len(_S), (len(_T), len(_S))
_TABLE = str.maketrans(_T, _S)


def norm(s):
    return (s or "").translate(_TABLE).lower()


import re as _re

# Phrases Whisper is known to produce on silence / wind / music (whole-segment matches only)
_HALLUCINATION = _re.compile(
    r"^\s*(you\.?|thank you\.?|thanks for watching[!.]*|thank you (so much )?for watching[!.]*|"
    r"please (like and )?subscribe.*|subscribe.*channel.*|eu me assino[!.]*|obrigad[oa] por assistir.*|"
    r"ご視聴ありがとうございました.*|字幕.*|.*amara\.org.*|请不吝点赞.*|請不吝點贊.*|感谢观看.*|感謝觀看.*|"
    r"谢谢观看.*|謝謝觀看.*|明镜与点点.*)\s*$", _re.I)


def clean_segments(segs):
    """Drop known recogniser hallucinations and segments flagged as non-speech by the model."""
    out = []
    for s in segs:
        if _HALLUCINATION.match(s.get("text", "")):
            continue
        if s.get("no_speech", 0) > 0.6 and s.get("logprob", 0) < -0.8:
            continue
        out.append(s)
    return out
