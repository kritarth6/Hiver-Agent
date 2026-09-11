# Amazon Customer Support AI Agent

An AI-powered customer support agent built using the Customer Support on Twitter (TWCS) dataset. The system focuses on AmazonHelp conversations and performs intent classification, historical-response retrieval, and human escalation.

## Problem Statement

Customer support teams receive a large number of repetitive customer queries. This project builds a support agent that can:

* Classify customer messages into support intents
* Retrieve relevant responses from historical conversations
* Detect situations that require human escalation
* Provide a simple interactive Streamlit interface

## Dataset

Dataset: Customer Support on Twitter (TWCS)

The dataset contains customer-support conversations between users and brands on Twitter.

For this project, the `AmazonHelp` brand was selected because it had a large number of customer-support interactions.

### Conversation Reconstruction

Customer messages and AmazonHelp responses were connected using:

* `in_response_to_tweet_id`
* `response_tweet_id`

This produced **70,956 valid AmazonHelp customer-support conversation pairs**.

## Intent Categories

The system uses 10 support intents:

1. `delivery_issue`
2. `return_issue`
3. `refund_issue`
4. `payment_issue`
5. `order_issue`
6. `product_issue`
7. `account_issue`
8. `technical_issue`
9. `contact_support`
10. `general_query`

## Machine Learning Baseline

A TF-IDF + Logistic Regression classifier was developed as the main baseline.

### Configuration

* TF-IDF features: 10,000
* N-grams: unigrams and bigrams
* Logistic Regression
* Maximum iterations: 1000

### Result

**Test Accuracy: 85.81%**

The model performed particularly well on intents such as:

* account issues
* delivery issues
* refund issues
* return issues
* order issues

The product issue class had lower recall because it had significantly fewer examples.

## Retrieval System

A TF-IDF retrieval index was created over the historical AmazonHelp customer messages.

For a new customer query:

1. The intent is identified.
2. Similar historical customer messages are searched.
3. The response associated with the most similar conversation is retrieved.
4. The retrieved response is presented as a suggested support response.

This approach allows the system to reuse responses from real historical customer-support interactions.

## Human Escalation

Some issues should not be handled automatically.

The system checks for escalation signals such as:

* urgent
* manager
* complaint
* several times
* multiple times
* not resolved
* not solved
* unacceptable

When detected, the system returns:

`ESCALATE_TO_HUMAN`

This prevents repeated or highly frustrated customer issues from being handled only through automated retrieval.

## Evaluation

A small functional test set containing six representative customer queries was used to verify the system.

The test cases covered:

* Delivery
* Refund
* Account
* Payment
* Product
* Human escalation

Result:

**6/6 correct = 100% functional test accuracy**

This is a small functional evaluation and should not be interpreted as a statistically representative model accuracy.

## Failure Analysis

One important retrieval failure occurred for a damaged-product query.

The system correctly identified the intent as:

`product_issue`

However, the retrieved historical response was not relevant to the customer's damaged-product complaint.

This demonstrates an important limitation of pure retrieval-based support: a semantically similar message may still have an inappropriate historical response.

Another limitation is that the initial intent labels were generated using keyword-based rules. Therefore, the 85.81% classifier result should be considered a baseline rather than a gold-standard supervised evaluation.

## Streamlit Demo

The project includes an interactive Streamlit application.

The application allows a user to:

1. Enter a customer message
2. Detect the support intent
3. Retrieve a historical support response
4. Escalate difficult or repeated complaints to a human

### Run the application

Activate the project virtual environment and run:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Project Structure

```text
hiver-support-agent/
│
├── app.py
├── README.md
│
├── data/
│   ├── raw/
│   │   └── twcs/
│   │       └── twcs.csv
│   │
│   └── processed/
│       ├── amazon_pairs.csv
│       └── amazon_eval.csv
│
└── notebooks/
    └── Notebook1
```

## Limitations

* Intent labels were initially generated using keyword-based rules.
* The functional evaluation set is small.
* Retrieval can return an incorrect historical response.
* The current demo does not generate completely new responses using an LLM.
* Escalation is based on predefined signals.

## Future Improvements

Possible improvements include:

* Human-labeled golden evaluation data
* Sentence-transformer embeddings for better semantic retrieval
* LLM-based response generation
* LLM-as-a-judge evaluation
* Confidence-based escalation
* Better conversation reconstruction
* More robust intent classification
* Monitoring and feedback from real support agents

## Technologies

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Streamlit
* Natural Language Processing

## Conclusion

This project demonstrates an end-to-end AI customer-support workflow using real customer-support conversations. It combines intent classification, historical-response retrieval, and human escalation into an interactive support-agent prototype.
