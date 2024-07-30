from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("LogisticRegressionModel.pkl", 'rb'))
df = pd.read_csv("cleaned df.csv")


@app.route('/')
def index():
    gender = sorted(df['Gender'].unique())
    education = sorted(df['Educational Qualification '].unique())
    income = sorted(df['income'].unique())
    family_type = sorted(df['Family type'].unique())
    location = sorted(df[' Location'].unique())
    reside = sorted(df['reside'].unique())
    occupation = sorted(df['Occupation'].unique())
    marital_status = sorted(df['marital_status'].unique())
    mother_tongue = sorted(df['Mother tongue'].unique())
    save_time = sorted(df[' Save time'].unique())
    detailed_product_information = sorted(df['Detailed product information'].unique())
    visual_demonstrations_in_video_reviews = sorted(df['Visual demonstrations in video reviews'].unique())
    trust_in_reputable_reviewers = sorted(df['Trust in reputable reviewers'].unique())
    recommendations_from_like_minded_individuals = sorted(df['Recommendations from like-minded individuals'].unique())
    engaging_with_a_community_of_reviewers = sorted(df['Engaging with a community of reviewers'].unique())
    availability_of_a_wide_range_of_information = sorted(df['Availability of a wide range of information'].unique())
    reviews_are_transparent_and_authentic = sorted(df['Reviews are transparent and authentic'].unique())
    assurance_of_customer_satisfaction = sorted(df['Assurance of customer satisfaction'].unique())
    confirmation_of_product_suitability_for_specific_needs = sorted(df['Confirmation of product suitability for specific needs'].unique())
    confidence_in_my_knowledge_about_the_product = sorted(df['Confidence in my knowledge about the product '].unique())
    lack_of_interest_in_others_opinion = sorted(df['Lack of interest in others opinion'].unique())
    confidence_in_my_own_decision_making_abilities = sorted(df['Confidence in my own decision making abilities'].unique())
    prefer_seek_advice_from_experts = sorted(df['Prefer seek advice from experts'].unique())
    review_creates_confusion = sorted(df['Review creates confusion'].unique())
    difference_in_ewom_and_product_quality = sorted(df['Difference in eWOM and product quality'].unique())

    return render_template('index.html', gender=gender, education=education, income=income, family_type=family_type, location=location, reside=reside, occupation=occupation, marital_status=marital_status, mother_tongue=mother_tongue, save_time=save_time, detailed_product_information=detailed_product_information, visual_demonstrations_in_video_reviews=visual_demonstrations_in_video_reviews, trust_in_reputable_reviewers=trust_in_reputable_reviewers, recommendations_from_like_minded_individuals=recommendations_from_like_minded_individuals, engaging_with_a_community_of_reviewers=engaging_with_a_community_of_reviewers, availability_of_a_wide_range_of_information=availability_of_a_wide_range_of_information, reviews_are_transparent_and_authentic=reviews_are_transparent_and_authentic, assurance_of_customer_satisfaction=assurance_of_customer_satisfaction, confirmation_of_product_suitability_for_specific_needs=confirmation_of_product_suitability_for_specific_needs, confidence_in_my_knowledge_about_the_product=confidence_in_my_knowledge_about_the_product, lack_of_interest_in_others_opinion=lack_of_interest_in_others_opinion, confidence_in_my_own_decision_making_abilities=confidence_in_my_own_decision_making_abilities, prefer_seek_advice_from_experts=prefer_seek_advice_from_experts, review_creates_confusion=review_creates_confusion, difference_in_ewom_and_product_quality=difference_in_ewom_and_product_quality)


@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    education = request.form.get('education')
    income = request.form.get('income')
    family_type = request.form.get('family_type')
    location = request.form.get('location')
    reside = request.form.get('reside')
    occupation = request.form.get('occupation')
    dependent = int(request.form.get('dependent'))
    marital_status = request.form.get('marital_status')
    mother_tongue = request.form.get('mother_tongue')
    save_time = request.form.get('save_time')
    detailed_product_information = request.form.get('detailed_product_information')
    visual_demonstrations_in_video_reviews = request.form.get('visual_demonstrations_in_video_reviews')
    trust_in_reputable_reviewers = request.form.get('trust_in_reputable_reviewers')
    recommendations_from_like_minded_individuals = request.form.get('recommendations_from_like_minded_individuals')
    engaging_with_a_community_of_reviewers = request.form.get('engaging_with_a_community_of_reviewers')
    availability_of_a_wide_range_of_information = request.form.get('availability_of_a_wide_range_of_information')
    reviews_are_transparent_and_authentic = request.form.get('reviews_are_transparent_and_authentic')
    assurance_of_customer_satisfaction = request.form.get('assurance_of_customer_satisfaction')
    confirmation_of_product_suitability_for_specific_needs = request.form.get('confirmation_of_product_suitability_for_specific_needs')
    confidence_in_my_knowledge_about_the_product = request.form.get('confidence_in_my_knowledge_about_the_product')
    lack_of_interest_in_others_opinion = request.form.get('lack_of_interest_in_others_opinion')
    confidence_in_my_own_decision_making_abilities = request.form.get('confidence_in_my_own_decision_making_abilities')
    prefer_seek_advice_from_experts = request.form.get('prefer_seek_advice_from_experts')
    review_creates_confusion = request.form.get('review_creates_confusion')
    difference_in_ewom_and_product_quality = request.form.get('difference_in_ewom_and_product_quality')

    print(age, gender, education, income, family_type, location, reside, occupation, dependent, marital_status, mother_tongue, save_time, detailed_product_information, visual_demonstrations_in_video_reviews, trust_in_reputable_reviewers, recommendations_from_like_minded_individuals, engaging_with_a_community_of_reviewers, availability_of_a_wide_range_of_information, reviews_are_transparent_and_authentic, assurance_of_customer_satisfaction, confirmation_of_product_suitability_for_specific_needs, confidence_in_my_knowledge_about_the_product, lack_of_interest_in_others_opinion, confidence_in_my_own_decision_making_abilities, prefer_seek_advice_from_experts, review_creates_confusion, difference_in_ewom_and_product_quality)
    input = pd.DataFrame([[age, gender, education, income, family_type, location, reside, occupation, dependent, marital_status, mother_tongue, save_time, detailed_product_information, visual_demonstrations_in_video_reviews, trust_in_reputable_reviewers, recommendations_from_like_minded_individuals, engaging_with_a_community_of_reviewers, availability_of_a_wide_range_of_information, reviews_are_transparent_and_authentic, assurance_of_customer_satisfaction, confirmation_of_product_suitability_for_specific_needs, confidence_in_my_knowledge_about_the_product, lack_of_interest_in_others_opinion, confidence_in_my_own_decision_making_abilities, prefer_seek_advice_from_experts, review_creates_confusion, difference_in_ewom_and_product_quality]], columns=['age', 'Gender', 'Educational Qualification ', 'income', 'Family type', ' Location', 'reside', 'Occupation', 'dependent', 'marital_status', 'Mother tongue', ' Save time', 'Detailed product information', 'Visual demonstrations in video reviews', 'Trust in reputable reviewers', 'Recommendations from like-minded individuals', 'Engaging with a community of reviewers', 'Availability of a wide range of information', 'Reviews are transparent and authentic', 'Assurance of customer satisfaction', 'Confirmation of product suitability for specific needs', 'Confidence in my knowledge about the product ', 'Lack of interest in others opinion', 'Confidence in my own decision making abilities', 'Prefer seek advice from experts', 'Review creates confusion', 'Difference in eWOM and product quality'])
    prediction = model.predict(input)[0]

    return str(prediction)


if __name__ == '__main__':
    app.run(debug=True)
