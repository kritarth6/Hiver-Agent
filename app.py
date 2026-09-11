import streamlit as st
import pandas as pd
import re
from collections import Counter

st.set_page_config(
    page_title="Amazon Support Agent",
    page_icon="🤖"
)

st.title("🤖 Amazon Customer Support Agent")
st.write("AI-powered Amazon customer support using intent classification and historical response retrieval.")

# Load processed Amazon conversations
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/amazon_pairs.csv")

amazon_pairs = load_data()


# Intent classification
def assign_intent(text):

    text = text.lower()

    if any(word in text for word in ["refund", "refunded", "money back"]):
        return "refund_issue"

    elif any(word in text for word in ["return", "returns", "returning"]):
        return "return_issue"

    elif any(word in text for word in [
        "delivery", "delivered", "shipping",
        "package", "parcel", "courier"
    ]):
        return "delivery_issue"

    elif any(word in text for word in [
        "payment", "paid", "charge",
        "charged", "card"
    ]):
        return "payment_issue"

    elif any(word in text for word in [
        "damaged", "broken", "defective", "faulty"
    ]):
        return "product_issue"

    elif any(word in text for word in [
        "login", "log in", "sign in",
        "account", "password", "membership"
    ]):
        return "account_issue"

    elif any(word in text for word in [
        "app", "website", "site",
        "error", "not working"
    ]):
        return "technical_issue"

    elif any(word in text for word in [
        "agent", "executive",
        "customer service", "call"
    ]):
        return "contact_support"

    elif any(word in text for word in [
        "order", "ordered", "ordering"
    ]):
        return "order_issue"

    else:
        return "general_query"


# Escalation detection
def should_escalate(query):

    query = query.lower()

    escalation_words = [
        "urgent",
        "escalate",
        "manager",
        "complaint",
        "still not",
        "again",
        "multiple times",
        "several times",
        "not resolved",
        "not solved",
        "nobody has solved",
        "very frustrated",
        "unacceptable"
    ]

    return any(word in query for word in escalation_words)


# Simple text similarity
def similarity(query, text):

    query_words = set(
        re.findall(r"\b\w+\b", query.lower())
    )

    text_words = set(
        re.findall(r"\b\w+\b", text.lower())
    )

    if not query_words or not text_words:
        return 0

    common_words = query_words.intersection(text_words)

    return len(common_words) / len(query_words)


# Support agent
def support_agent(query):

    if should_escalate(query):
        return "ESCALATE_TO_HUMAN", None

    intent = assign_intent(query)

    # Search only conversations from the predicted intent
    candidates = amazon_pairs[
        amazon_pairs["intent"] == intent
    ]

    if len(candidates) == 0:
        candidates = amazon_pairs

    scores = candidates["customer_text"].apply(
        lambda text: similarity(query, text)
    )

    best_index = scores.idxmax()

    response = candidates.loc[
        best_index,
        "response_text"
    ]

    return intent, response


# User input
st.subheader("Customer Message")

query = st.text_area(
    "Describe your issue:",
    placeholder="Example: My package has not arrived yet.",
    height=120
)


if st.button("Get Support", type="primary"):

    if not query.strip():

        st.warning("Please enter a customer message.")

    else:

        intent, response = support_agent(query)

        st.divider()

        if intent == "ESCALATE_TO_HUMAN":

            st.error("⚠️ ESCALATION REQUIRED")

            st.write(
                "This issue should be handled by a human support agent."
            )

        else:

            st.success("Support analysis completed")

            st.write("### Detected Intent")

            st.info(intent)

            st.write("### Suggested Support Response")

            st.success(response)

            st.write("### System Decision")

            st.write(
                "The agent classified the customer's issue "
                "and retrieved a similar historical Amazon support response."
            )