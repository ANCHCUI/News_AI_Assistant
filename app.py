import streamlit as st
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# 1. 设置一个专属密码
MY_PASSWORD = "sirenbang"

# 2. 创建密码输入框
password_attempt = st.text_input("请输入访问密码：", type="password")

# 3. 判断逻辑
if password_attempt != MY_PASSWORD:
    st.warning("密码错误或未输入密码，请重试。")
    st.stop() # 密码不对？直接让程序在这里停下，不加载后面的内容！

# ======== 下面继续写你原来的 AI 助手代码 ========
st.success("密码正确，欢迎使用 AI 新闻助手！")
# ... (原来加载模型和预测的代码) ...


# 核心秘诀：使用 @st.cache_resource 缓存模型。
# 这样哪怕网页刷新，模型也只需要训练一次，不会每次点击都重新训练！
@st.cache_resource
def load_and_train_model():
    dataset = load_dataset("ag_news")
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, stop_words='english')),
        ('clf', LinearSVC(random_state=42))
    ])
    pipeline.fit(train_texts, train_labels)
    return pipeline


# ================= 网页界面设计 =================

# 1. 网页标题和说明
st.set_page_config(page_title="AI 新闻助手", page_icon="📰")
st.title("📰 AI 新闻助手")
st.markdown("这是一个基于传统机器学习 (TF-IDF + SVM) 的新闻分类系统。")

# 2. 加载模型（带加载动画）
with st.spinner("AI 正在启动并加载数据集（首次加载可能需要半分钟）..."):
    classifier = load_and_train_model()

# 3. 用户输入区
user_input = st.text_area("请输入一段新闻文本 (英文)：", height=150,
                          placeholder="例如：Apple announces the new iPhone with AI features...")

# 4. 按钮与预测逻辑
if st.button("🚀 开启 AI 智能预测", type="primary"):
    if user_input.strip() == "":
        st.warning("请先输入一些新闻内容哦！")
    else:
        # 进行预测
        categories = ["🌍 世界 (World)", "⚽ 体育 (Sports)", "💰 财经 (Business)", "💻 科技 (Sci/Tech)"]
        pred_idx = classifier.predict([user_input])[0]

        # 显示结果
        st.success("预测成功！")
        st.metric(label="AI 分类结果", value=categories[pred_idx])
        st.balloons()  # 预测成功放个气球特效！