# A knowledge-guided early-warning system for digital-transformation failure in entrepreneurial firms

::: {.frontmatter}
Divya Munagala
:::

## Abstract

Digital-transformation failure in an entrepreneurial firm is a decision, not only a rank. The warning has to catch a conjunction a committee can name, and the alert has to be chosen by a stated cost. This short communication presents KG-EWS, a knowledge-guided early-warning score. Eight committee rules turn the operational state into a signed evidence term and add it to the log-odds of a logistic score fit on noisy measurements. The evidence weight and the alert threshold are chosen on a validation slice to minimize a decision cost that charges a missed failure five times a false alarm. Across 120 reproducible cohorts of 380 ventures, KG-EWS reached a mean AUC of 0.765 and a mean decision cost of 0.598. Paired Wilcoxon tests favor it over logistic regression (cost advantage 0.024, 95% CI 0.008 to 0.040, p = 1.14×10⁻³; AUC advantage 0.022, p = 1.24×10⁻¹⁴), over gradient boosting, and over the rules used alone. KG-EWS is cheaper in 59 cohorts, more expensive in 28, and identical in 33. The gain remains when a missed failure is charged three times a false alarm, and it is not detectable when the charge is eight. It does not survive a coding error that flips each predicate with probability 0.15. Used alone, the rules catch rule-triggered failures only by alarming on 92.2% of surviving firms, so the issued warning is the combined score and the names of the rules that fired. The benchmark is synthetic and does not establish a calendar lead time.

## Keywords

knowledge-based system; early warning; digital transformation; entrepreneurship; innovation; decision cost; venture data

## 1. Introduction

An entrepreneurial firm can invest in a digital offer and still fail when capability, team experience, or innovation throughput does not arrive. Founders already describe those failures as rules. Early-warning research runs from discriminant ratios for corporate bankruptcy [1] to machine-learning scores for the same task [2]. Reviews show that model choice and the definition of distress both change the reported accuracy [3,4]. A financial early-warning logit has been specified and checked for whether its alerts are efficient [5]. Machine-learning scores have also been placed in a decision-support system for small-firm crisis [6].

Digital-transformation research describes the gap when investment runs ahead of operating skill [7,8] and treats the change as multidisciplinary [9,10]. The venture has to turn a technology into a repeatable business [11,12] with capabilities it can reconfigure [13]. Smaller-firm studies connect that change to digital orientation and capability [14], persistence [15], strategy configuration [16], and venture-level transformation capability [17]. Mapping treats digital transformation and innovation as joint themes [18]. Distress research returns a probability [1–6]. Transformation research returns an account of capability [7–18]. The decision needs committee conjunctions kept visible, and an alert chosen by a stated cost.

A noisy linear score can assign a modest probability to high investment with low capability, innovation, and team experience. Knowledge-based decision support writes that conjunction into the score [19,20]. Image and text bankruptcy models can separate failed firms without a contestable rule [21,22], and the retained fields also change a distress score [23].

KG-EWS charges each missed failure five times each false alarm, adds eight production rules to logistic log-odds with the weight chosen on validation data, and tests that score on 120 cohorts in which the rules see the operational state and the statistical models see only noisy measurements. The same program re-selects the weight at two other miss costs and corrupts the predicates. The warning is a probability and the names of the rules that fired. Expert weights are not the failure-process coefficients, and no failure date is generated.

## 2. KG-EWS

### 2.1 Decision cost

Let y = 1 if a venture fails and y = 0 otherwise. An alert is issued when score s meets threshold τ. In a cohort of n firms, FN is the number of failures with no alert and FP is the number of surviving firms alerted. The decision cost is

::: {.equation}
J = (5 FN + FP) / n. (1)
:::

The weight 5 is a stated preference, not a loss estimated from an investor. Alerting every firm costs about the share of firms that survive. One avoided miss offsets as many as five added false alarms. The same J chooses τ for every method, on the validation slice only. A rule-triggered failure is a failed firm for which at least one risk rule in Section 2.2 fires. A smooth-stress failure is a failed firm for which none fires. Miss rates in those two groups are descriptive. They are not extra terms in Eq. (1).

### 2.2 Knowledge-rule layer

Eight production rules fire only when every part of the antecedent holds on the operational state. Seven raise the failure evidence. P1 is protective.

- **R1.** Digital investment above 0.62 and capability below 0.38.
- **R2.** Digital investment above 0.58 and innovation throughput below 0.28.
- **R3.** Digital investment above 0.52 and team digital experience below 0.32.
- **R4.** Runway under 5.5 months and burn multiple above 2.2.
- **R5.** Customer concentration above 0.72 and market turbulence above 0.58.
- **R6.** At least three pivots and adoption lag above 0.65.
- **R7.** Team digital experience below 0.28, innovation throughput below 0.32, and digital investment above 0.48.
- **P1.** Capability above 0.68, innovation throughput above 0.58, and runway above 14 months.

Let zᵢ = 1 when rule i fires and 0 otherwise. The evidence is

::: {.equation}
e = Σᵢ cᵢ zᵢ, (2)
:::

with expert weights c = (1.00, 0.85, 0.80, 0.95, 0.60, 0.55, 0.75, −1.05) in the order R1 through P1. The weights record direction and a rough magnitude. They are not fitted, and they are not the coefficients in Section 3. The construction follows the idea that pieces of evidence can be modular contributions to a belief [20], written here as a linear term so that λ in Eq. (3) stays visible.

### 2.3 The warning score

The measured-field model is a class-balanced logistic regression on ten noisy measurements: runway, burn multiple, digital investment, capability, innovation throughput, team digital experience, market turbulence, customer concentration, pivot count, and adoption lag. Predictors are standardized on the fit slice. Let p_Lin be its probability. KG-EWS adds the rule evidence on the log-odds scale,

::: {.equation}
s = σ( logit(p_Lin) + λ e ), (3)
:::

where σ is the logistic function and λ ≥ 0 is chosen from the grid {0, 0.2, …, 3.0} to minimize validation J. The alert threshold τ is then chosen on that same slice. If the rules do not help, the minimum is at λ = 0 and Eq. (3) returns the logistic score. Gradient boosting [24] on the same noisy measurements is a baseline. The experiment asks whether measurement noise is enough to make the explicit rules worth adding.

**Algorithm 1. KG-EWS**

Input: noisy measurements x; committee predicates z; fit, validation, and test index sets. Output: test scores s and an alert threshold τ.

1. Fit the logistic model and, separately, the gradient-boosting baseline on the fit slice.
2. Compute evidence e = Σ cᵢ zᵢ for every venture.
3. For each λ on the grid, form s = σ(logit(p_Lin) + λ e) on the validation slice and choose τ to minimize J.
4. Keep the λ whose best validation J is smallest, and freeze that λ and its τ.
5. Return s and the alerts on the test slice, together with the identities of the rules that fired.

## 3. Experimental Design

The benchmark is synthetic. Cohort k uses seed 9000 + k. There are 120 cohorts of 380 ventures, split into a fit slice (52.5%), a validation slice (17.5%), and a test slice (30%), stratified on the failure label. Models are fit once. λ and τ are frozen before the test slice is scored. Runway is uniform on 1 to 24 months. Burn is lognormal. Digital investment, capability, innovation, team experience, turbulence, concentration, and adoption lag are beta draws. Pivot count is Poisson. Failure is a Bernoulli draw with log-odds

::: {.equation}
η = −2.15 + 0.16(11 − runway) + 0.85(burn − 1.2) + 1.35 turbulence + 1.10 concentration + 1.70 R1 + 1.40 R2 + 1.20 R3 + 1.50 R4 + 1.00 R5 + 0.80 R6 + 1.15 R7 − 1.55 P1 + ε, (4)
:::

where ε is normal with mean 0 and standard deviation 0.35. The mean failure rate is 32.6%. Of those failures, 35.0% are rule-triggered. The statistical models see a noisy copy: runway noise of scale 0.80 months, burn noise of scale 0.15, noise of scale 0.18 on the digital, team, and adoption-lag fields, noise of scale 0.04 on turbulence and concentration, and noise of scale 0.70 on the pivot count. The rules are coded on the clean operational state.

Four scores are compared on every test slice: logistic regression, gradient boosting [24], the rule evidence passed through its own logistic map, and KG-EWS. Logistic regression is the measured model without the rules. The knowledge base is the rules without the measured model. KG-EWS is both. Cohort-level costs and AUCs are paired. The test is a one-sided Wilcoxon signed-rank test [25] of whether KG-EWS has lower cost, and separately higher AUC, than each baseline, with a 95% t interval for the mean paired difference. On the same 120 cohorts, λ and τ are selected again at miss weights 3 and 8. Separately, 60 cohorts with seeds 12000 through 12059 flip each of the eight predicate bits independently with probability 0.15 before the evidence is summed. That probability was fixed before those cohorts were generated.

## 4. Results and Discussion

Table 1 is the mean across 120 test slices. The 95% interval for the mean cost is 0.579 to 0.617 for KG-EWS and 0.602 to 0.642 for logistic regression. Those intervals nearly touch, so Figure 1 plots the paired difference inside each cohort. The trade between missed rule-triggered failures and false alarms among surviving firms is in Table 1.

| Method | AUC | Decision cost | Rule-miss | Smooth-miss | False-alarm rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.743 | 0.622 | 0.118 | 0.137 | 0.614 |
| Knowledge base | 0.645 | 0.633 | 0.004 | 0.012 | 0.922 |
| Gradient boosting | 0.696 | 0.675 | 0.117 | 0.142 | 0.687 |
| KG-EWS | 0.765 | 0.598 | 0.067 | 0.194 | 0.536 |

: **Table 1.** Mean test-slice performance over 120 cohorts.

Decision cost is Eq. (1). Rule-miss is the share of rule-triggered failures with no alert. Smooth-miss is the share of other failures with no alert. The false-alarm rate is among surviving firms. Lower is better for cost, miss rates, and false alarms. Higher is better for AUC.

![](fig1_results.png)

**Figure 1.** Paired reduction in decision cost, baseline minus KG-EWS, plotted for each of 120 cohorts. The larger marker is the mean, with a 95% t interval. A point above the zero line favors KG-EWS.

KG-EWS is the only row in Table 1 that leads on both AUC and cost. Because the mean-cost intervals nearly touch, the claim is the paired comparison in Figure 1 and Table 2. The interval against logistic regression lies above zero. Logistic regression already fits the additive part of Eq. (4). The 0.024 cost reduction is the gain from the conjunctions that measurement noise obscures. The reductions of 0.035 and 0.077 are mostly gaps against a weaker instrument.

The knowledge-base rule-miss of 0.004 is not a usable policy: that score alarms on 92.2% of surviving firms. Case-based distress prediction [26] and rough-set with Dempster–Shafer fusion [27] are different constructions. KG-EWS adds one rule block to one logistic score, the role of a financial early-warning logit [5]. Gradient boosting is 0.047 below logistic regression and 0.069 below KG-EWS in AUC [21,22]. Relative to logistic regression, rule-miss falls from 0.118 to 0.067 and the false-alarm rate from 0.614 to 0.536, while smooth-miss rises from 0.137 to 0.194. λ was positive in 90 of 120 cohorts (mean 0.78, SD 0.81). The stored case is the largest lift among failed, rule-triggered firms in cohorts with λ > 0 and a lift of at least 0.08. R1, R2, R3, and R7 fire (evidence 3.40). The scores are 0.118, 0.016, 0.971, and 0.999. The warning names the four rules. Digital-capability studies are the setting for that argument [14–17].

Removing the measured model moves cost from 0.598 to 0.633 and AUC from 0.765 to 0.645. Removing the rules raises cost by 0.024 and lowers AUC by 0.022. For a founder watching digital capability [7,8,14], the rules are the part of the warning that can be argued in words, and the smaller share of the gain.

| Comparison | Metric | Advantage | 95% CI | Better / worse / tied | p-value |
| --- | ---: | ---: | --- | ---: | ---: |
| Versus logistic regression | Decision cost | 0.024 | 0.008 to 0.040 | 59 / 28 / 33 | 1.14×10⁻³ |
| Versus logistic regression | AUC | 0.022 | 0.017 to 0.027 | 84 / 6 / 30 | 1.24×10⁻¹⁴ |
| Versus gradient boosting | Decision cost | 0.077 | 0.056 to 0.099 | 91 / 28 / 1 | 2.08×10⁻¹⁰ |
| Versus gradient boosting | AUC | 0.069 | 0.061 to 0.076 | 117 / 3 / 0 | 1.30×10⁻²¹ |
| Versus knowledge base | Decision cost | 0.035 | 0.017 to 0.054 | 81 / 35 / 4 | 2.91×10⁻⁵ |
| Versus knowledge base | AUC | 0.121 | 0.112 to 0.130 | 119 / 1 / 0 | 2.37×10⁻²¹ |

: **Table 2.** Paired comparisons on the 120 cohort-level test results.

The advantage is baseline cost minus KG-EWS cost, or KG-EWS AUC minus baseline AUC. The p-value is a one-sided Wilcoxon signed-rank test [25]. Of the 33 logistic cost ties, 30 set λ to 0. Of the 90 cohorts with λ > 0, KG-EWS was cheaper in 59, more expensive in 28, and tied in 3. Mean gaps were 0.086 and −0.080. AUC then rises in 84 of those 90 cohorts and falls in 6.

Bankruptcy models for listed firms have been ranked by the financial cost of failure and by jobs lost [28]. Eq. (1) is narrower: a fixed five-to-one charge, stated in advance. At miss weight 3 the mean test cost was 0.493 for KG-EWS and 0.523 for logistic regression (advantage 0.030, 95% CI 0.019 to 0.041, cheaper in 70 cohorts, p = 6.14×10⁻⁷). Mean λ rose to 1.01 and was positive in 100 cohorts. At weight 8 the mean costs were 0.685 and 0.681 (advantage −0.004, 95% CI −0.026 to 0.018, cheaper in 52 cohorts, more expensive in 31, tied in 37, p = 0.135).

On the 60 cohorts with corrupted antecedents, the paired cost advantage over logistic regression is −0.015 (95% CI −0.036 to 0.006, cheaper in 18, more expensive in 24, tied in 18, p = 0.892). The AUC advantage is −0.004 (95% CI −0.012 to 0.004, p = 0.527). λ stays positive in 42 of 60 cohorts (mean 0.46). The corrupted rules alone have mean AUC 0.578 and mean cost 0.722. KG-EWS remains cheaper than that rule score by 0.082 (p = 1.19×10⁻⁴). Rule-miss is 0.128 for KG-EWS and 0.130 for logistic regression. Under a correct 5:1 coding, the warning still alerts on 53.6% of surviving firms. Survey studies of digital strategy [15–17] describe the setting. They do not supply this alert.

## 5. Limitations

At a miss weight of 8 the paired advantage is compatible with zero. The two neighboring weights are not a full map of the loss surface. The 0.15 predicate flip removes the advantage over logistic regression, and λ stayed positive in most of those cohorts, so validation is an incomplete protection. The generator uses the same antecedents as the knowledge base. The expert weights differ from the data-generating coefficients, but the thresholds are the same thresholds. This is a recovery benchmark for a stated rule set.

There is no clock. Dynamic distress models for listed companies reweight older statements when the failure process drifts [29]. A firm here is one draw. Nothing here is a lead time. The cohorts are not a sample of industries, countries, or funding regimes. Field records with a failure date, a feature cutoff before that date, and predicates coded without knowledge of the outcome are the next test.

## 6. Conclusion

KG-EWS returns a probability and the names of the rules that fired. On 120 synthetic cohorts it has the lowest decision cost and the highest AUC of the four instruments. The gain over logistic regression is cheaper in 59 cohorts, more expensive in 28, and identical in 33. It remains at a 3:1 miss weight, is not detectable at 8:1, and disappears when each predicate is flipped with probability 0.15. The rules and the failure process share their thresholds. That is the claim.

## Data Availability

The cohorts are synthetic. `ews.py` generates the ventures, fits the models, chooses λ and τ, repeats that choice at miss weights 3 and 8, runs the 60-cohort predicate-flip check, writes `cohort_results.csv`, and redraws the figure. Primary seeds are 9000 through 9119. Stress seeds are 12000 through 12059. The program and the cohort table are at https://github.com/divzmunagala03-tech/Dba. No external data file is required.

## Declaration of Competing Interest

The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Cursor to draft language, organize the manuscript, and structure the code. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article. The experimental results, tables, and figure were produced by executing `ews.py`.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT Authorship Contribution Statement

**Divya Munagala:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Writing – original draft, Writing – review and editing, Visualization.

## References

::: {.refs}
[1] Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. The Journal of Finance, 23(4), 589–609. https://doi.org/10.2307/2978933

[2] Barboza, F., Kimura, H., & Altman, E. (2017). Machine learning models and bankruptcy prediction. Expert Systems with Applications, 83, 405–417. https://doi.org/10.1016/j.eswa.2017.04.006

[3] Alaka, H. A., Oyedele, L. O., Owolabi, H. A., Kumar, V., Ajayi, S. O., Akinade, O. O., & Bilal, M. (2018). Systematic review of bankruptcy prediction models: Towards a framework for tool selection. Expert Systems with Applications, 94, 164–184. https://doi.org/10.1016/j.eswa.2017.10.040

[4] Sun, J., Li, H., Huang, Q.-H., & He, K.-Y. (2014). Predicting financial distress and corporate failure: A review from the state-of-the-art definitions, modeling, sampling, and featuring approaches. Knowledge-Based Systems, 57, 41–56. https://doi.org/10.1016/j.knosys.2013.12.006

[5] Li, S., & Wang, S. (2014). A financial early warning logit model and its efficiency verification approach. Knowledge-Based Systems, 70, 78–87. https://doi.org/10.1016/j.knosys.2014.03.017

[6] Perboli, G., & Arabnezhad, E. (2021). A machine learning-based DSS for mid and long-term company crisis prediction. Expert Systems with Applications, 174, 114758. https://doi.org/10.1016/j.eswa.2021.114758

[7] Vial, G. (2019). Understanding digital transformation: A review and a research agenda. The Journal of Strategic Information Systems, 28(2), 118–144. https://doi.org/10.1016/j.jsis.2019.01.003

[8] Warner, K. S. R., & Wäger, M. (2019). Building dynamic capabilities for digital transformation: An ongoing process of strategic renewal. Long Range Planning, 52(3), 326–349. https://doi.org/10.1016/j.lrp.2018.12.001

[9] Verhoef, P. C., Broekhuizen, T., Bart, Y., Bhattacharya, A., Dong, J. Q., Fabian, N., & Haenlein, M. (2021). Digital transformation: A multidisciplinary reflection and research agenda. Journal of Business Research, 122, 889–901. https://doi.org/10.1016/j.jbusres.2019.09.022

[10] Kraus, S., Durst, S., Ferreira, J. J., Veiga, P., Kailer, N., & Weinmann, A. (2022). Digital transformation in business and management research: An overview of the current status quo. International Journal of Information Management, 63, 102466. https://doi.org/10.1016/j.ijinfomgt.2021.102466

[11] Nambisan, S. (2017). Digital entrepreneurship: Toward a digital technology perspective of entrepreneurship. Entrepreneurship Theory and Practice, 41(6), 1029–1055. https://doi.org/10.1111/etap.12254

[12] Nambisan, S., Wright, M., & Feldman, M. (2019). The digital transformation of innovation and entrepreneurship: Progress, challenges and key themes. Research Policy, 48(8), 103773. https://doi.org/10.1016/j.respol.2019.03.018

[13] Teece, D. J. (2007). Explicating dynamic capabilities: The nature and microfoundations of (sustainable) enterprise performance. Strategic Management Journal, 28(13), 1319–1350. https://doi.org/10.1002/smj.640

[14] Rupeika-Apoga, R., Petrovska, K., & Bule, L. (2022). The effect of digital orientation and digital capability on digital transformation of SMEs during the COVID-19 pandemic. Journal of Theoretical and Applied Electronic Commerce Research, 17(2), 669–685. https://doi.org/10.3390/jtaer17020035

[15] Christofi, M., Khan, H., Zahoor, N., Hadjielias, E., & Tarba, S. (2024). Digital transformation of SMEs: The role of entrepreneurial persistence and market sensing dynamic capability. IEEE Transactions on Engineering Management, 71, 13598–13615. https://doi.org/10.1109/TEM.2022.3230248

[16] Wang, H., & Wu, W. (2024). The effects of digital technology strategy configurations on new venture performance. IEEE Transactions on Engineering Management, 71, 5470–5486. https://doi.org/10.1109/TEM.2024.3363625

[17] Kumar, V., Kumar, S., Chatterjee, S., & Mariani, M. (2024). Optimizing the digital transformation capability for enhancing economic sustainability of entrepreneurial venture: The moderating role of entrepreneurial orientation. IEEE Transactions on Engineering Management, 71, 8517–8530. https://doi.org/10.1109/TEM.2024.3387540

[18] Gao, P., Wu, W., & Yang, Y. (2022). Discovering themes and trends in digital transformation and innovation research. Journal of Theoretical and Applied Electronic Commerce Research, 17(3), 1162–1184. https://doi.org/10.3390/jtaer17030059

[19] Chen, H., Chiang, R. H. L., & Storey, V. C. (2012). Business intelligence and analytics: From big data to big impact. MIS Quarterly, 36(4), 1165–1188. https://doi.org/10.2307/41703503

[20] Shortliffe, E. H., & Buchanan, B. G. (1975). A model of inexact reasoning in medicine. Mathematical Biosciences, 23(3–4), 351–379. https://doi.org/10.1016/0025-5564(75)90047-4

[21] Hosaka, T. (2019). Bankruptcy prediction using imaged financial ratios and convolutional neural networks. Expert Systems with Applications, 117, 287–299. https://doi.org/10.1016/j.eswa.2018.09.039

[22] Mai, F., Tian, S., Lee, C., & Ma, L. (2019). Deep learning models for bankruptcy prediction using textual disclosures. European Journal of Operational Research, 274(2), 743–758. https://doi.org/10.1016/j.ejor.2018.10.024

[23] Liang, D., Tsai, C.-F., & Wu, H.-T. (2015). The effect of feature selection on financial distress prediction. Knowledge-Based Systems, 73, 289–297. https://doi.org/10.1016/j.knosys.2014.10.010

[24] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

[25] Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83. https://doi.org/10.2307/3001968

[26] Li, H., & Sun, J. (2008). Ranking-order case-based reasoning for financial distress prediction. Knowledge-Based Systems, 21(8), 868–878. https://doi.org/10.1016/j.knosys.2008.03.047

[27] Xiao, Z., Yang, X., Pang, Y., & Dang, X. (2012). The prediction for listed companies’ financial distress by using multiple prediction methods with rough set and Dempster–Shafer evidence theory. Knowledge-Based Systems, 26, 196–206. https://doi.org/10.1016/j.knosys.2011.08.001

[28] Radovanovic, J., & Haas, C. (2023). The evaluation of bankruptcy prediction models based on socio-economic costs. Expert Systems with Applications, 227, 120275. https://doi.org/10.1016/j.eswa.2023.120275

[29] Sun, J., Fujita, H., Chen, P., & Li, H. (2017). Dynamic financial distress prediction with concept drift based on time weighting combined with Adaboost support vector machine ensemble. Knowledge-Based Systems, 120, 4–14. https://doi.org/10.1016/j.knosys.2016.12.019
:::
