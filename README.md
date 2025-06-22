# Modern Portfolio Theory & Optimization Analysis
*Advanced Quantitative Methods and Machine Learning in Finance*

## **Problem Statement**

Modern Portfolio Theory (MPT) faces practical implementation challenges when constructing optimal portfolios across multiple asset classes. Investment managers must navigate complex trade-offs between expected returns, risk exposure, and diversification benefits while satisfying various portfolio constraints. The challenge is to develop robust optimization frameworks that can identify efficient portfolios, including the Global Minimum Variance (GMV) portfolio and Maximum Sharpe Ratio (MSR) portfolio, using real-world industry data. This analysis addresses the fundamental question of how to optimally allocate capital across diverse industry sectors to achieve specific risk-return objectives.

## **Abstract**

This project implements comprehensive Modern Portfolio Theory optimization techniques using the Fama-French 48 Industry Portfolios dataset spanning January 2002 onwards. The analysis constructs and evaluates efficient portfolios through three distinct approaches: multi-asset efficient frontier analysis, Global Minimum Variance portfolio optimization, and Maximum Sharpe Ratio portfolio construction. Using ten carefully selected industries representing diverse economic sectors, the study employs constrained optimization algorithms to identify optimal asset allocation strategies. The research demonstrates practical applications of mean-variance optimization, risk-return trade-offs, and the mathematical foundations underlying efficient portfolio construction for institutional investment management.

## **Dataset Description**

The analysis utilizes the Fama-French 48 Industry Portfolios dataset, focusing on monthly value-weighted returns for a carefully curated selection of ten industries representing diverse economic sectors.

**Dataset Specifications:**
- **Temporal Coverage**: January 2002 onwards (recent market data for contemporary relevance)
- **Industry Selection**: 10 strategically chosen sectors from 48 available industries
- **Return Measurement**: Monthly value-weighted portfolio returns
- **Data Format**: Decimal returns (converted from percentage format)
- **Primary Source**: [Kenneth R. French Data Library, Dartmouth College](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- **Direct Dataset Link**: [48 Industry Portfolios (Value-Weighted Returns)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/48_Industry_Portfolios_CSV.zip)

**Selected Industry Portfolio Composition:**
1. **Soda**: Soft drinks and beverage manufacturing
2. **Meals**: Restaurants and food service establishments  
3. **Agric**: Agriculture and farming operations
4. **Fin**: Financial services and banking
5. **Insur**: Insurance companies and related services
6. **Books**: Publishing and media companies
7. **Clths**: Clothing and textile manufacturing
8. **Trans**: Transportation and logistics services
9. **Smoke**: Tobacco products and related industries
10. **Beer**: Beer and alcoholic beverage production

This selection ensures broad economic sector representation, combining defensive industries (Insurance, Utilities-like sectors), consumer staples (Food, Beverages), cyclical sectors (Transportation, Agriculture), and discretionary industries (Clothing, Entertainment).

## **Methodology & Analysis Framework**

### **Data Preprocessing Protocol**

The research implements comprehensive data preparation procedures:
- Systematic loading and validation of Fama-French industry portfolio data
- Conversion from percentage to decimal return format for optimization compatibility
- Temporal filtering to focus on post-2002 data for contemporary market relevance
- Expected return calculation using geometric mean approach: `((1 + ret).prod()) ** (1/n_years) - 1`
- Covariance matrix computation with annualization factor: `ret.cov() * 12`

### **Portfolio Optimization Framework**

The study implements three distinct optimization approaches using constrained optimization algorithms:

#### **1. Multi-Asset Efficient Frontier Construction**

```python
def portfolio_ret(weights, returns):
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights)**0.5

def optimal_weights(n_points, ereturns, covmat):
    target_returns = np.linspace(ereturns.min(), ereturns.max(), n_points)
    opt_weights = [gmv(tr, ereturns, covmat) for tr in target_returns]
    return opt_weights
```

The efficient frontier analysis constructs 50 optimal portfolios across the risk-return spectrum, illustrating the fundamental trade-off between expected returns and portfolio volatility.

**EFFICIENT FRONTIER** 

![image](https://github.com/user-attachments/assets/be4df664-d4ed-4830-93c6-4bb1536a7fe0)


#### **2. Global Minimum Variance Portfolio Optimization**

```python
def gmv(ereturns, covmat):
    n = len(ereturns)
    init_values = np.repeat(1/n, n)
    bounds = ((0,1),)*n
    
    weights_sum_to_1 = {'type': 'eq',
                       'fun': lambda weights: np.sum(weights)-1}
    
    results = minimize(portfolio_vol,
                      init_values,
                      args=(covmat,),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=(weights_sum_to_1))
    
    return results.x
```

The GMV optimization solves the constrained minimization problem:
```
minimize: σ²p = w'Σw
subject to: Σwi = 1, wi ≥ 0
```

#### **3. Maximum Sharpe Ratio Portfolio Construction**

```python
def msr(ereturns, covmat, riskfree_rate=0):
    def neg_sharpe_ratio(weights):
        ret = portfolio_ret(weights, ereturns)
        vol = portfolio_vol(weights, covmat)
        return -(ret - riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio,
                      init_values,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=(weights_sum_to_1))
    
    return results.x
```

The MSR optimization maximizes the Sharpe ratio by minimizing its negative:
```
maximize: (μp - rf)/σp
equivalent to minimize: -(μp - rf)/σp
subject to: Σwi = 1, wi ≥ 0
```

## **Results & Portfolio Analysis**

### **Efficient Frontier Characteristics**

The efficient frontier analysis reveals the classic hyperbolic relationship between risk and return across the ten-industry portfolio universe. The frontier demonstrates:

- **Risk Range**: Portfolio volatility spans from approximately 15.4% (minimum variance) to higher levels approaching individual asset volatilities
- **Return Range**: Expected returns range from defensive levels to growth-oriented targets
- **Optimal Portfolios**: Each point represents the maximum return achievable for a given risk level
- **Trade-off Dynamics**: The concave shape illustrates diminishing marginal returns to risk-taking

**Interpretation**: The efficient frontier validates Modern Portfolio Theory principles, showing that diversification across industries provides superior risk-return combinations compared to individual asset investments. The lower portion of the frontier contains suboptimal portfolios where increased risk doesn't significantly enhance returns, while the upper region represents efficient portfolios maximizing return per unit of risk.

### **Global Minimum Variance Portfolio Results**

**Optimal Asset Allocation:**
| **Industry** | **Weight** | **Allocation Strategy** |
|-------------|------------|------------------------|
| **Insur** | 45.18% | Dominant defensive allocation |
| **Beer** | 33.60% | Significant consumer staples position |
| **Smoke** | 13.55% | Moderate defensive exposure |
| **Soda** | 6.26% | Small consumer discretionary allocation |
| **Agric** | 1.40% | Minimal cyclical exposure |
| **Others** | 0.00% | Excluded for risk minimization |

**Portfolio Characteristics:**
- **Expected Return**: 13.17% annually
- **Portfolio Volatility**: 15.40% annually (minimum achievable)
- **Risk Reduction**: Significant diversification benefits through selective allocation

**Analysis**: The GMV portfolio demonstrates a clear preference for defensive sectors, with Insurance and Beer dominating the allocation. The complete exclusion of Meals, Finance, Books, Clothes, and Transportation indicates these sectors contribute excessive risk relative to their diversification benefits. The concentration in traditionally stable industries reflects the optimization's focus on risk minimization rather than return maximization.

### **Maximum Sharpe Ratio Portfolio Results**

**Optimal Asset Allocation:**
| **Industry** | **Weight** | **Strategic Focus** |
|-------------|------------|-------------------|
| **Insur** | 48.58% | Primary risk-adjusted return driver |
| **Beer** | 21.58% | Secondary consumer staples allocation |
| **Smoke** | 19.97% | Defensive high-return contributor |
| **Soda** | 9.87% | Complementary consumer exposure |
| **Others** | 0.00% | Excluded for suboptimal risk-return profile |

**Portfolio Characteristics:**
- **Expected Return**: 13.69% annually
- **Portfolio Volatility**: 15.61% annually
- **Sharpe Ratio**: 0.8766 (superior risk-adjusted performance)
- **Risk-Return Efficiency**: Optimal point on the efficient frontier

**Analysis**: The MSR portfolio exhibits remarkable concentration in just four industries, with Insurance maintaining dominance at 48.58%. The inclusion of defensive consumer staples (Beer, Smoke, Soda) alongside Insurance creates an optimal risk-adjusted return profile. The Sharpe ratio of 0.8766 indicates exceptional efficiency in generating excess returns per unit of risk. This concentration suggests that the excluded industries fail to provide adequate compensation for their risk contributions.

### **Comparative Portfolio Analysis**

| **Metric** | **GMV Portfolio** | **MSR Portfolio** | **Analysis** |
|------------|------------------|------------------|--------------|
| **Expected Return** | 13.17% | 13.69% | MSR achieves higher returns |
| **Volatility** | 15.40% | 15.61% | Minimal volatility difference |
| **Sharpe Ratio** | 0.8545 | 0.8766 | MSR superior risk-adjusted performance |
| **Diversification** | 5 industries | 4 industries | Both favor concentrated strategies |
| **Insurance Weight** | 45.18% | 48.58% | Insurance dominates both portfolios |

**Key Insights:**
1. **Sectoral Concentration**: Both optimal portfolios heavily favor defensive sectors, particularly Insurance
2. **Risk-Return Trade-off**: The MSR portfolio achieves 3.9% higher returns for only 1.4% additional volatility
3. **Diversification Paradox**: Optimal portfolios concentrate in fewer assets, contradicting naive diversification
4. **Consumer Staples Premium**: Beverage and tobacco industries provide superior risk-adjusted returns

## **Theoretical Framework & Implementation**

### **Mathematical Foundations**

**Portfolio Return:**
```
μp = Σ(wi × μi)
where wi = weight of asset i, μi = expected return of asset i
```

**Portfolio Variance:**
```
σ²p = w'Σw = Σ Σ(wi × wj × σij)
where Σ = covariance matrix, σij = covariance between assets i and j
```

**Sharpe Ratio:**
```
SR = (μp - rf)/σp
where rf = risk-free rate (assumed zero in this analysis)
```

### **Optimization Constraints**

**Budget Constraint:**
```
Σwi = 1 (portfolio weights sum to 100%)
```

**Non-negativity Constraint:**
```
wi ≥ 0 ∀i (no short selling allowed)
```

**Upper Bound Constraint:**
```
wi ≤ 1 ∀i (maximum 100% allocation per asset)
```

## **Implementation Guide**

### **Technical Requirements**

```bash
# Required Dependencies
pip install numpy pandas matplotlib scipy scikit-learn
```

### **Data Loading & Preprocessing**

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load and preprocess data
ret = pd.read_csv("48_Industry_Portfolios1.csv", header=0, index_col=0)
ret = ret.apply(pd.to_numeric, errors='coerce') / 100

# Calculate annualized metrics
n_years = ret.shape[0] / 12
eret = ((1 + ret).prod()) ** (1/n_years) - 1
cov = ret.cov() * 12
```

### **Portfolio Optimization Workflow**

**Step 1: Efficient Frontier Construction**
1. Define target return range across minimum to maximum expected returns
2. Solve constrained optimization for each target return level
3. Generate portfolio weights, returns, and volatilities
4. Plot efficient frontier visualization

**Step 2: Global Minimum Variance Portfolio**
1. Implement GMV optimization function with volatility minimization objective
2. Apply budget constraint (weights sum to 1) and non-negativity constraints
3. Calculate optimal weights and portfolio characteristics
4. Analyze sector allocation and risk reduction benefits

**Step 3: Maximum Sharpe Ratio Portfolio**
1. Implement MSR optimization by minimizing negative Sharpe ratio
2. Apply identical constraints as GMV optimization
3. Calculate optimal weights and risk-adjusted performance metrics
4. Compare with GMV portfolio for strategic insights

## **Investment Implications & Conclusions**

### **Strategic Investment Insights**

**Sector Rotation Strategy**: The analysis reveals that Insurance and consumer staples (Beer, Tobacco, Beverages) historically provided superior risk-adjusted returns during the study period. This suggests defensive sector rotation strategies may outperform broad market diversification.

**Concentration vs. Diversification**: Contrary to naive diversification approaches, both optimal portfolios concentrate holdings in 4-5 industries, indicating that effective diversification requires selective asset allocation rather than equal weighting across all available options.

**Risk Management Applications**: The GMV portfolio provides a blueprint for risk-averse institutional investors seeking stable returns with minimal volatility, while the MSR portfolio offers guidance for performance-oriented strategies targeting maximum risk-adjusted returns.

### **Limitations & Considerations**

**Sample Period Bias**: The analysis covers January 2002 onwards, potentially reflecting specific market conditions that may not persist in future periods.

**Industry Classification**: Fama-French industry definitions may not capture evolving business models and cross-sector convergence in modern markets.

**Transaction Costs**: The optimization framework assumes costless rebalancing, which may not reflect real-world implementation constraints.

**Parameter Stability**: Expected returns and covariances are assumed constant, though these parameters exhibit significant time variation in practice.

## **Future Research Directions**

### **Model Extensions**
- **Dynamic Portfolio Optimization**: Implement time-varying parameter models for evolving market conditions
- **Factor Model Integration**: Incorporate Fama-French factor loadings for enhanced risk attribution
- **Transaction Cost Models**: Include realistic trading costs and turnover constraints
- **Regime-Switching Models**: Account for different market states and structural breaks

### **Alternative Optimization Approaches**
- **Black-Litterman Framework**: Integrate investor views with market equilibrium assumptions
- **Robust Optimization**: Address parameter uncertainty through worst-case scenario optimization
- **Multi-Objective Optimization**: Balance multiple investment criteria beyond risk-return trade-offs
- **Machine Learning Integration**: Apply ML techniques for improved return prediction and risk modeling

---

*This research demonstrates practical applications of Modern Portfolio Theory optimization techniques to real-world industry data, providing actionable insights for institutional portfolio management and strategic asset allocation decisions.*
