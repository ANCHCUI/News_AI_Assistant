import streamlit as st
from transformers import pipeline

import streamlit as st
from transformers import pipeline

# ==========================================
# 🔒 第一道防线：密码拦截区
# ==========================================
# type="password" 会让输入的内容变成小黑点保护隐私
pwd = st.text_input("请输入专属访问密码 🔑：", type="password")

# 这里设置你的密码
if pwd != "SHU1234":
    if pwd != "":  # 如果用户输入了东西但不对，给个报错提示
        st.error("密码错误，禁止访问！🛑")

    # 【核心魔法】：密码不对，立刻停止！后面的深度学习模型根本不会被唤醒
    st.stop()

# ==========================================
# 🔓 密码正确后，才会执行下面的深度学习代码
# ==========================================
st.success("密码正确，欢迎长官！")


@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


st.title("🚀 深度学习新闻分析大脑")
st.write("告别死板的规则！你随便定几个标签，AI 都能凭借常识自己完成分类。")


# ... (保留你之前剩下的所有代码，不用改) ...
# 1. 缓存加载模型（这步超级关键，防止网页每次刷新都重新下载几十MB的模型）
@st.cache_resource
def load_model():
    # 这里我们借用一个支持中文的轻量级深度学习预训练模型
    return pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


# 正在加载模型的提示
with st.spinner("正在唤醒深度学习模型...（首次加载约需1分钟，请耐心等待）"):
    classifier = load_model()

# 2. 设置界面输入框
default_text = "最新的人工智能大模型在自然语言处理领域取得了重大突破。同时，关于如何通过宪法和相关法规来规范AI技术的讨论，也成为了近期多边外交与国际关系的重要议题。"
user_input = st.text_area("📝 输入一篇新闻或文章段落：", default_text, height=150)

# 让用户自己定义标签
labels_input = st.text_input("🏷️ 设定你想要的分类标签（用英文逗号隔开）：", "人工智能, 国际政治, 足球赛事, 基础科学")

# 3. 运行分析
if st.button("开始深度理解"):
    if user_input and labels_input:
        with st.spinner("AI 正在阅读字里行间的语义..."):
            # 把用户输入的标签用逗号切分成列表
            labels = [label.strip() for label in labels_input.split(",")]

            # 把文章和标签扔给模型
            result = classifier(user_input, labels)

            st.success("分析完成！请看 AI 给出的置信度：")

            # 4. 把 AI 给出的概率用进度条直观地画出来
            for label, score in zip(result['labels'], result['scores']):
                st.write(f"**{label}**: {score * 100:.1f}%")
                st.progress(float(score))
    else:
        st.warning("文章和标签都不能为空哦！")