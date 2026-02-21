
Cell 1: Setup and Data Generation
This cell will import necessary libraries and create our hypothetical dataset for EU countries, including GDP, inflation, and a classification 'Y' or 'N'.
import pandas as pd
import numpy as np

# --- 1. Setup Data ---

# List of hypothetical EU countries (for demonstration)
countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
    "Slovenia", "Spain", "Sweden"
]

# Generate random nominal GDP for 2025 (in billions USD)
# Let's make it somewhat realistic, ranging from ~50B to ~4000B
np.random.seed(42) # for reproducibility
gdp_2025 = np.random.uniform(50, 4000, len(countries))
gdp_2025[gdp_2025 < 100] *= 0.5 # A few smaller countries
gdp_2025 = np.round(gdp_2025, 2)

# Generate random inflation rate for 2025 (as percentage, e.g., 1.5 for 1.5%)
# Ranging from -1% to 15% (some outliers to show scoring behavior)
inflation_2025 = np.random.uniform(-1, 15, len(countries))
inflation_2025 = np.round(inflation_2025, 2)

# Generate random classification (Y or N) for each country
classification = np.random.choice(['Y', 'N'], len(countries))

# Create a Pandas DataFrame
df = pd.DataFrame({
    'Country': countries,
    'GDP_2025_BUSD': gdp_2025,
    'Inflation_2025_Pct': inflation_2025,
    'Classification': classification
})

print("Raw Data Sample:")
print(df.head())
print("\nData Info:")
df.info()

Cell 2: Define Weights and Scoring Parameters
Here, we'll define the different sets of weights based on the 'Y' or 'N' classification, and parameters for our attribute scoring functions.
# --- 2. Define Weights and Scoring Parameters ---

# Weights for GDP (w1) and Inflation (w2)
# The order of weights must match the order of attribute scores in the score matrix
# For 'Y' classified countries: GDP is weighted higher
weights_Y = np.array([0.65, 0.35]) # 65% GDP, 35% Inflation

# For 'N' classified countries: Inflation is weighted higher
weights_N = np.array([0.40, 0.60]) # 40% GDP, 60% Inflation

# Scoring parameters for GDP
# Higher GDP is better. Score will be 1 (min_gdp) to 100 (max_gdp)
# We'll use the actual min/max from the data for scaling

# Scoring parameters for Inflation
# Ideal inflation target (e.g., ECB target)
IDEAL_INFLATION_TARGET = 2.0 # percentage
# Maximum deviation from target that maps to a score of 1 (worst)
# E.g., if inflation is 12% (10% deviation from 2%) or -8% (10% deviation), score is 1.
MAX_INFLATION_DEV_FOR_MIN_SCORE = 10.0 # percentage points

print("Weights for 'Y' countries (GDP, Inflation):", weights_Y)
print("Weights for 'N' countries (GDP, Inflation):", weights_N)
print(f"Inflation Scoring: Ideal {IDEAL_INFLATION_TARGET}%, Max deviation for min score {MAX_INFLATION_DEV_FOR_MIN_SCORE}%")

Cell 3: Attribute Scoring Functions
We'll define two functions: one for scoring GDP and one for scoring inflation. These functions will normalize/scale the raw values to a 1-100 range.
# --- 3. Attribute Scoring Functions ---

def score_gdp(gdp_values: np.ndarray) -> np.ndarray:
    """
    Scores GDP values on a scale of 1 to 100. Higher GDP gets a higher score.
    Uses linear scaling based on the min and max GDP in the provided array.
    """
    min_gdp = gdp_values.min()
    max_gdp = gdp_values.max()

    if max_gdp == min_gdp: # Handle case where all GDPs are the same
        return np.full_like(gdp_values, 50.0) # Assign a neutral score

    # Linear scaling: 1 + 99 * (value - min) / (max - min)
    gdp_scores = 1 + 99 * (gdp_values - min_gdp) / (max_gdp - min_gdp)
    return np.round(gdp_scores, 2)

def score_inflation(inflation_values: np.ndarray) -> np.ndarray:
    """
    Scores inflation values on a scale of 1 to 100. Scores are higher closer
    to the IDEAL_INFLATION_TARGET. Values far from the target get lower scores.
    """
    # Calculate absolute deviation from the ideal target
    deviation = np.abs(inflation_values - IDEAL_INFLATION_TARGET)

    # Scale deviation to score reduction
    # A deviation of MAX_INFLATION_DEV_FOR_MIN_SCORE leads to a 99-point reduction from 100 (i.e., score of 1)
    score_reduction = (deviation / MAX_INFLATION_DEV_FOR_MIN_SCORE) * 99

    # Calculate initial score
    inflation_scores = 100 - score_reduction

    # Clamp scores between 1 and 100
    inflation_scores = np.maximum(1, np.minimum(100, inflation_scores))
    return np.round(inflation_scores, 2)

print("Scoring functions defined.")

Cell 4: Calculate Attribute Scores for Each Country
Apply the scoring functions to our DataFrame columns and store the results in new columns.
# --- 4. Calculate Attribute Scores ---

# Convert DataFrame columns to numpy arrays for efficient processing
gdp_data = df['GDP_2025_BUSD'].to_numpy()
inflation_data = df['Inflation_2025_Pct'].to_numpy()

# Calculate scores
df['GDP_Score'] = score_gdp(gdp_data)
df['Inflation_Score'] = score_inflation(inflation_data)

print("Attribute Scores Calculated:")
print(df[['Country', 'GDP_2025_BUSD', 'GDP_Score', 'Inflation_2025_Pct', 'Inflation_Score', 'Classification']].head())

Cell 5: Prepare for Matrix Arithmetic - Weights and Masks
This is where we construct the combined weights matrix based on the classification.
# --- 5. Prepare for Matrix Arithmetic ---

# Extract the attribute scores into a NumPy array
# The order must match the order of weights (GDP score, Inflation score)
scores_matrix = df[['GDP_Score', 'Inflation_Score']].to_numpy()

# Create a boolean mask for 'Y' classified countries
is_Y_country = (df['Classification'] == 'Y').to_numpy()

# Initialize a weights matrix with zeros, shape (num_countries, num_attributes)
# This matrix will hold the specific weights for each country
weights_combined_matrix = np.zeros_like(scores_matrix)

# Apply weights_Y to 'Y' countries using the mask
# NumPy's broadcasting will assign weights_Y (1D array) to selected rows (2D slice)
weights_combined_matrix[is_Y_country] = weights_Y

# Apply weights_N to 'N' countries (the inverse of the 'Y' mask)
weights_combined_matrix[~is_Y_country] = weights_N

print("Scores Matrix (first 5 rows):")
print(scores_matrix[:5])
print("\nCombined Weights Matrix (first 5 rows):")
print(weights_combined_matrix[:5])
print(f"\nShape of scores_matrix: {scores_matrix.shape}")
print(f"Shape of weights_combined_matrix: {weights_combined_matrix.shape}")

Cell 6: Final Score Calculation using Matrix Arithmetic
Perform the element-wise multiplication and then sum across the attributes to get the final scorecard for each country.
# --- 6. Final Score Calculation using Matrix Arithmetic ---

# Perform element-wise multiplication of scores and weights
# This gives us [GDP_Score * w1, Inflation_Score * w2] for each country
weighted_scores = scores_matrix * weights_combined_matrix

# Sum across the columns (axis=1) to get the final combined score for each country
# The result is a 1D array of total scores
final_scores = np.sum(weighted_scores, axis=1)

# Ensure final scores are capped between 1 and 100 (though they should be by design)
final_scores = np.maximum(1.0, np.minimum(100.0, final_scores))
final_scores = np.round(final_scores, 2)

# Add the final scores to the DataFrame
df['Final_Score'] = final_scores

print("Final Scorecard (Top 10 countries by score):")
print(df[['Country', 'GDP_Score', 'Inflation_Score', 'Classification', 'Final_Score']].sort_values(by='Final_Score', ascending=False).head(10))

print("\nFinal Scorecard (Bottom 10 countries by score):")
print(df[['Country', 'GDP_Score', 'Inflation_Score', 'Classification', 'Final_Score']].sort_values(by='Final_Score', ascending=True).head(10))

This set of snippets provides a complete workflow within a Python notebook, starting from data generation, defining scoring logic, implementing conditional weights using a mask, and finally calculating the scorecard using matrix arithmetic with NumPy, all while leveraging Pandas DataFrames for structured data handling.



import pandas as pd
import numpy as np

# Sample data for 2025 (Projected/Dummy values)
countries = ['Germany', 'France', 'Italy', 'Spain', 'Poland']
data = {
    'GDP_Nominal': [4500, 3100, 2200, 1600, 850], # in Billions
    'Inflation_Rate': [2.1, 2.4, 1.9, 3.2, 4.5]   # in %
}

df_stats = pd.DataFrame(data, index=countries)

# Masking data: Y/N classification
# This determines which weight set a country receives
mask_labels = pd.Series(['Y', 'Y', 'N', 'N', 'Y'], index=countries, name='Group')

print("Economic Data:")
print(df_stats)

def normalize_scores(df):
    norm_df = pd.DataFrame(index=df.index)
    
    # GDP: Higher value = Higher score
    norm_df['Score_GDP'] = 1 + 99 * (df['GDP_Nominal'] - df['GDP_Nominal'].min()) / \
                           (df['GDP_Nominal'].max() - df['GDP_Nominal'].min())
    
    # Inflation: Lower value = Higher score
    norm_df['Score_Inf'] = 1 + 99 * (df['Inflation_Rate'].max() - df['Inflation_Rate']) / \
                           (df['Inflation_Rate'].max() - df['Inflation_Rate'].min())
    return norm_df

df_scores = normalize_scores(df_stats)

# Define weight sets
# Set Y: Focuses more on GDP | Set N: Focuses more on Inflation
weights_Y = [0.7, 0.3]
weights_N = [0.4, 0.6]

# Map the mask to the actual weights
# We create a (Countries x Attributes) weight matrix
weight_matrix = np.array([weights_Y if label == 'Y' else weights_N for label in mask_labels])

print("Weight Matrix (First 3 rows):")
print(weight_matrix[:3])


# Convert scores to numpy for matrix math
score_array = df_scores.to_numpy()

# Element-wise multiplication followed by horizontal summation
final_scores = np.sum(score_array * weight_matrix, axis=1)

# Back into a clean DataFrame
df_final = pd.DataFrame({
    'Group': mask_labels,
    'Final_Score': final_scores
}, index=countries).sort_values(by='Final_Score', ascending=False)

print(df_final)