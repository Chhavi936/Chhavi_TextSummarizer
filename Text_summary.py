
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Initialize NLTK modules
nltk.download('punkt')
nltk.download('stopwords')

raw_text = """A nutrient is a substance used by an organism to survive, grow, and reproduce. The requirement for dietary nutrient intake applies to animals, plants, fungi, and protists. Nutrients can be incorporated into cells for metabolic purposes or excreted by cells to create non-cellular structures, such as hair, scales, feathers, or exoskeletons. Some nutrients can be metabolically converted to smaller molecules in the process of releasing energy, such as for carbohydrates, lipids, proteins, and fermentation products (ethanol or vinegar), leading to end-products of water and carbon dioxide. All organisms require water. Essential nutrients for animals are the energy sources, some of the amino acids that are combined to create proteins, a subset of fatty acids, vitamins and certain minerals. Plants require more diverse minerals absorbed through roots, plus carbon dioxide and oxygen absorbed through leaves. Fungi live on dead or living organic matter and meet nutrient needs from their host.
Different types of organisms have different essential nutrients. Ascorbic acid (vitamin C) is essential, meaning it must be consumed in 
sufficient amounts, to humans and some other animal species, but some animals and plants are able to synthesize it. Nutrients may be 
organic or inorganic: organic compounds include most compounds containing carbon, while all other chemicals are inorganic. Inorganic 
nutrients include nutrients such as iron, selenium, and zinc, while organic nutrients include, among many others, energy-providing
compounds and vitamins.A classification used primarily to describe nutrient needs of animals divides nutrients into macronutrients and
micronutrients. Consumed in relatively large amounts (grams or ounces), macronutrients (carbohydrates, fats, proteins, water) are primarily used to 
generate energy or to incorporate into tissues for growth and repair. Micronutrients are needed in smaller amounts (milligrams or 
micrograms); they have subtle biochemical and physiological roles in cellular processes, like vascular functions or nerve conduction.
Inadequate amounts of essential nutrients, or diseases that interfere with absorption, result in a deficiency state that compromises 
growth, survival and reproduction. Consumer advisories for dietary nutrient intakes, such as the United States Dietary Reference 
Intake, are based on deficiency outcomes[clarification needed] and provide macronutrient and micronutrient guides for both lower and 
upper limits of intake. In many countries, macronutrients and micronutrients in significant content[clarification needed] are required 
by regulations to be displayed on food product labels. Nutrients in larger quantities than the body needs may have harmful effects.
Edible plants also contain thousands of compounds generally called phytochemicals which have unknown effects on disease or health,
including a diverse class with non-nutrient status called polyphenols, which remain poorly understood as of 2017."""


def summarizer(raw_text):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(raw_text)
    
    # Initialize stop words and stemmer
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    
    # Compute the word frequencies in the paragraph
    word_frequencies = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            word = word.lower()
            if word not in stop_words:
                stem_word = stemmer.stem(word)
                if stem_word in word_frequencies:
                    word_frequencies[stem_word] += 1
                else:
                    word_frequencies[stem_word] = 1
    
    # Calculate the sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            word = word.lower()
            if word not in stop_words:
                stem_word = stemmer.stem(word)
                if stem_word in word_frequencies:
                    if sentence in sentence_scores:
                        sentence_scores[sentence] += word_frequencies[stem_word]
                    else:
                        sentence_scores[sentence] = word_frequencies[stem_word]
    
    # Determine the average sentence score
    total_score = sum(sentence_scores.values())
    average_score = total_score / len(sentence_scores)
    
    # Generate the summary by selecting sentences with scores above the average
    summary = [sentence for sentence, score in sentence_scores.items() if score > average_score]
    summary = ' '.join(summary)
    
    return summary,raw_text,len(raw_text.split(' ')) ,len(summary.split(' '))


