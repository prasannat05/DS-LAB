https://chatgpt.com/share/697abfd9-e024-800d-a4da-dafae0b010e6 
https://chatgpt.com/share/697abfd9-e024-800d-a4da-dafae0b010e6 untill exp 5
1️⃣ revenue

Type: Numerical (Continuous)

Meaning: Annual income of the applicant

Typical Range: ₹20,000 – ₹500,000+ (depends on dataset)

Higher Value → Lower credit risk (generally)

2️⃣ dti_n (Debt-to-Income Ratio)

Type: Numerical (Continuous)

Meaning: Monthly debt obligations divided by monthly income (excluding mortgage)

Valid Realistic Range: 0 – 50%

Ideal Range:

0–20 → Low risk

20–35 → Moderate

35–50 → High risk

Higher Value → Higher default probability

3️⃣ loan_amnt

Type: Numerical (Continuous)

Meaning: Loan amount requested by borrower

Typical Range: 1,000 – 40,000 (depends on Lending Club data year)

Higher Loan → Slightly higher financial burden

4️⃣ fico_n

Type: Numerical (Continuous)

Meaning: Credit score of borrower

Valid FICO Range: 300 – 850

Your Dataset Range: 612 – 847 (✔ Valid & Good)

Risk Interpretation:

750+ → Excellent

700–749 → Good

650–699 → Fair

Below 650 → Risky

Higher FICO → Lower default risk

5️⃣ experience_c

Type: Numerical (Binary/Indicator)

Meaning: Employment stability indicator

Values:

1 → Experienced / Stable

0 → Less stable

6️⃣ purpose

Type: Categorical

Meaning: Purpose of loan

Common Categories:

debt_consolidation

credit_card

home_improvement

medical

small_business

personal

etc.

7️⃣ home_ownership_n

Type: Categorical

Meaning: Housing status

Categories:

RENT

OWN

MORTGAGE

OTHER

8️⃣ addr_state

Type: Categorical

Meaning: State of borrower

Values: US state codes (CA, TX, NY, FL, etc.)

Used for geographical risk analysis

9️⃣ issue_year

Type: Numerical (Discrete)

Meaning: Year loan was issued

Range Example: 2007 – 2018

Used for trend analysis

🔟 emp_length_n

Type: Numerical (Discrete)

Meaning: Employment length converted to numeric

Range:

0 → <1 year

1–9 → Years

10 → 10+ years

-1 → Unknown

1️⃣1️⃣ loan_repaid

Type: Binary (Target Variable)

Meaning: Loan repayment status

Values:

1 → Loan Repaid Successfully

0 → Loan Defaulted / Not Repaid

Used as dependent variable in classification models.
