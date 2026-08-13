Forecasting Household Appliance Energy Use

Benchmark, SARIMAX, feature-based and foundation models over a 24-hour
horizon

GitHub repository:
https://github.com/spage191089/appliance-energy-forecasting

1. Introduction

Forecasting how much energy a household's appliances will use over the
next day is useful for smart-home applications such as scheduling
flexible appliances, sizing battery storage and responding to
time-of-use tariffs. Forecasting at the level of an individual household
is particularly challenging because appliance use is influenced by
occupant behaviour. At the same time, household demand is not entirely
unpredictable: recurring daily and weekly routines create strong
temporal structure, while short periods of high consumption introduce
substantial variation around those patterns.

This report asks whether increasingly complex methods actually improve
on simple benchmarks when forecasting appliance energy use 24 hours
ahead. Five benchmark forecasts are evaluated first, followed by a
SARIMAX model, feature-based gradient-boosting models, and a
Chronos-Bolt foundation model applied zero-shot. Throughout, the
emphasis is not only on which model wins on average, but on whether an
apparent win can be trusted given how little test data is available.

The data are the UCI Appliances Energy Prediction dataset, recorded in a
low-energy house in Stambruges, Belgium over four and a half months from
January to May 2016 (Candanedo, Feldheim and Deramaix, 2017). Appliance
energy use was logged every 10 minutes, giving 19,735 readings,
alongside indoor temperature and humidity from a ZigBee sensor network
in nine rooms and outdoor weather from the airport station at Chièvres.
The target variable is Appliances, measured in watt-hours (Wh).

2. Data and preprocessing

The 10-minute grid was checked and is complete: no gaps, no duplicated
timestamps and no missing values, so no imputation was required. This
was verified by assertion rather than assumed.

Three preprocessing decisions were made. First, the series was
aggregated to hourly means. This removes the high-frequency switching
noise that dominates the 10-minute data, keeps the daily profile intact,
and reduces the series to 3,289 hourly observations, which is what makes
a 147-model SARIMAX search practical. The mean rather than the sum was
used so that energy, temperature and humidity are treated the same way;
since the hourly mean is proportional to the hourly total, this choice
does not change any accuracy comparison.

Second, incomplete hours were dropped: the final hour held only one
10-minute reading rather than six, so it was not comparable with the
rest of the series. Third, two columns named rv1 and rv2, which the
original authors added as random noise for a feature-selection exercise,
were removed so they could not leak into any feature set.

The resulting hourly target has a mean of 97.7 Wh and a maximum of 608
Wh.

One data-quality point is worth documenting: T6 and RH_6 are named
as if they were indoor sensors, but they are actually mounted outside on
the north-facing wall of the house. T6 falls as low as −5.9 °C and
correlates at 0.976 with outdoor temperature, so both variables were
treated as weather variables throughout the analysis.

Item                                Specification

Target                              Appliance energy use (Wh)

Original frequency                  10 minutes

Working frequency                   Hourly mean

Forecast horizon                    24 hours

Test period                         Final 14 days / 336 hours
(2016-05-13 18:00 to 2016-05-27
17:00)

Training observations               2,953 hourly observations

3. Exploratory analysis

Hourly appliance use is strongly right-skewed: the median is 63.3 Wh,
but the 95th percentile is 276.7 Wh. A log transform makes the
distribution nearly symmetric, but all models here were fitted on the
original scale so that MAE, RMSE and bias stay interpretable in
watt-hours. One consequence is that RMSE is pulled around by a small
number of high-load hours, which is why MAE and MASE are treated as the
primary accuracy measures in this report.

The most important feature of the series is its daily rhythm. Demand is
low overnight, climbs through the morning, and reaches a clear evening
peak. The individual daily profiles show that the timing of the pattern
is highly stable, but the height of individual evening peaks varies by a
factor of several. This combination of a stable shape and a volatile
amplitude explains a great deal of what follows.

An MSTL decomposition splits the series into seasonal and remainder
components. It attributes 37.5% of variance to the 24-hour seasonal
component, 16.0% to the 168-hour weekly component, and 1.4% to trend,
leaving 45.2% as remainder. This is not a hard ceiling on accuracy,
since a non-linear model could still find structure in that remainder,
but it shows plainly that a large share of the variation is not
explained by regular seasonality alone.

Two complementary stationarity tests were run. The ADF test rejects a
unit root on the level series (statistic −9.08, p < 0.001), and the
KPSS test does not reject stationarity (0.041, p ≥ 0.10), so no ordinary
differencing was applied (Kwiatkowski et al., 1992). The autocorrelation
function still shows clear dependence at lags 24 and 168, which is the
direct motivation for the seasonal benchmarks and for the seasonal term
in SARIMAX.

Correlations between appliance use and the environmental variables are
modest, and several weaken sharply once the average hour-of-day profile
is removed from both sides: outdoor temperature falls from 0.124 to
−0.019, and outdoor humidity from −0.193 to 0.040. In other words, much
of the apparent predictive information in weather and indoor sensors is
really just a proxy for the time of day, not an independent signal.

4. Forecasting design

The target is hourly appliance energy use and the horizon is 24 hours.
The final 14 days form the test period: 336 test hours and 2,953
training hours. Every model is judged against the strongest benchmark,
not just against the naive forecast, and all models are scored on MAE,
RMSE, MASE and bias.

A 24-hour horizon and a 14-day test period are not automatically the
same experiment, and the difference matters. Forecasting the whole
336-hour test block from a single starting point would really be a
14-day-ahead forecast, not a 24-hour one. To avoid this, a
rolling-origin design was used: the forecast origin moves forward one
day at a time and the training history grows with it, producing 14
separate 24-hour forecast windows (Hyndman and Athanasopoulos, 2021).

All three possible evaluation protocols were checked, and the rankings
differ enough between them that any claim of beating the benchmark is
only meaningful once the horizon and protocol are stated clearly.

MASE is scaled by the in-sample MAE of a daily seasonal naive forecast
on the training data, 53.36 Wh (Hyndman and Koehler, 2006), fixed once
and reused for every model so that all MASE values are directly
comparable.

Fourteen windows is a small sample, so every headline comparison also
carries a block bootstrap confidence interval on the difference in MAE,
built by resampling whole 24-hour windows rather than individual hours.
This separates a model that looks numerically better from one that is
measurably better.

Forecast-origin realism was treated carefully. Hour-of-day and
day-of-week variables are known in advance, so they can be used freely,
and lags of the target at 24 hours or more are also known at a 24-hour
horizon. Future weather and indoor sensor readings would not be
available in a real deployment, so any model using realised future
values of those variables is reported separately as a conditional
forecast rather than a genuine deployable one.

Leakage was controlled by scaling only on the training set, shifting the
target before any rolling feature was computed, and running a test that
corrupts every value after the forecast origin and checks that the
feature table does not change.

5. Benchmark models

Five simple benchmarks were evaluated: mean, naive, daily seasonal
naive, weekly seasonal naive and drift. The daily seasonal naive repeats
the value from 24 hours earlier; the weekly seasonal naive repeats the
value from 168 hours earlier.

Model                     MAE (Wh)   RMSE (Wh)    MASE   Bias (Wh)

Weekly seasonal naive        42.92       79.67   0.804      −12.62
Daily seasonal naive         48.39       85.79   0.907       +1.67
Mean                         50.01       73.99   0.937       −3.06
Naive                       143.38      170.62   2.687     +114.67
Drift                       143.94      171.26   2.698     +115.31

The weekly seasonal naive is the strongest benchmark, at MASE 0.804. Any
rule that carries a seasonal profile clearly beats the mean, naive and
drift forecasts, which confirms that recurring household routine is a
major source of predictable information.

The failure of the naive and drift forecasts has a specific, visible
cause: forecast origins fall at 17:00, near the evening peak, so both
carry a value of roughly 232 Wh through the entire following night.

The weekly method's advantage over the daily one is not statistically
resolved (−5.47 Wh, 95% interval [−18.96, +8.75]). The safest
conclusion is that seasonality itself matters a great deal, rather than
that weekly repetition is definitively better than daily repetition.

Empirical prediction intervals were also attached to this benchmark,
built from the quantiles of in-sample seasonal naive errors computed
separately for each hour of the day. Because they make no distributional
assumption, they widen during the volatile evening and narrow overnight;
they achieved 85.1% coverage at a nominal 80% level.

6. SARIMAX model

The required search covered non-seasonal orders p = 0--6, d = 0--2 and q
= 0--6: 147 combinations, of which 99 converged, taking 38.9 minutes to
fit.

The model with the lowest AIC overall, order (5,0,6), failed to
converge, so its AIC could not be used for selection. Restricting to the
99 models that did converge, SARIMA(1,0,6) was chosen by both AIC and
BIC. A secondary search selected a seasonal order of (0,1,1) at period
24; removing seasonal differencing costs 416.9 AIC units.

The final specification is:

SARIMA(1,0,6)(0,1,1)[24] with a constant

Residual diagnostics suggest the model has removed the linear serial
dependence in the series. The Ljung--Box test does not reject at lags
24, 48 or 168 (p = 0.72, 0.49, 0.15), and only 3 of the first 48
autocorrelation lags exceed the significance band.

Residual normality is weaker, with a skew of 1.88 and kurtosis of 11.25,
which matters when interpreting the model's Gaussian prediction
intervals.

Adding exogenous variables did not help. AIC rises from 32,137.6 for the
target-only model to 32,146.5 with calendar terms and 32,154.1 with
weather added on top, and no weather coefficient is statistically
significant.

The hourly Fourier terms are unidentified because seasonal differencing
at lag 24 removes deterministic functions with a period of exactly 24
hours. Day-of-week terms, with a period of 168 hours, survive this and
remain significant. In addition, T_out and Tdewpoint correlate at
0.792 and receive large, opposing coefficients with overlapping standard
errors, indicating multicollinearity.

SARIMAX specification            MAE (Wh)   RMSE (Wh)    MASE   Bias (Wh)

Target-only SARIMA                  36.79       64.21   0.689       −4.65
SARIMAX + calendar                  37.90       64.26   0.710       −4.99
SARIMAX + calendar + weather        38.37       64.74   0.719       −3.21

The target-only model improves on the weekly benchmark from MASE 0.804
to 0.689, a gain of 14.3%. The bootstrap interval on that improvement,
however, is [−17.54, +3.54] Wh, which includes zero.

The result supports the value of modelling the target's own temporal
dependence, but does not establish that SARIMAX reliably beats a
seasonal rule on this test period.

7. Feature-based model

A gradient-boosted tree regressor, implemented as scikit-learn's
HistGradientBoostingRegressor, was fitted to a feature table built
from lagged appliance use (1 to 168 hours back), rolling means and
standard deviations, calendar features and a weekend indicator, with
sensor and weather variables added in separate variants.

Hyperparameters were chosen using three-fold expanding-window
cross-validation on the training data only.

Several forecasting strategies were compared because a model built on
short lags can accidentally turn a 24-hour task into a one-hour-ahead
task. From an origin of 17:00, predicting 18:00 using a one-hour lag is
legitimate, but predicting 17:00 the next day using the same one-hour
lag would require a value 23 hours in the future.

Forecasting strategy                   MAE (Wh)   RMSE (Wh)    MASE   Bias (Wh)

Recursive                                 37.48       61.08   0.702       +0.21
Restricted (lags ≥ 24 h)                  38.56       65.76   0.723       −4.37
Direct multi-horizon (24 models)          38.59       65.27   0.723       −3.95
Covariates, lagged (operational)          40.12       64.93   0.752       −2.02
Covariates, realised (conditional)        45.98       68.21   0.862       +9.27

The recursive configuration is the strongest, at MASE 0.702, a 12.7%
improvement on the benchmark that is again not statistically resolvable
(bootstrap interval [−14.93, +3.04]).

Adding lagged environmental covariates makes the model worse, and using
realised future covariates makes it worse still. This strengthens the
finding that the covariates are weak because they genuinely carry little
information, not because the model fitting them was too simple.

One important diagnostic is the horizon trap. The identical model with
identical features reaches MASE 0.554 when evaluated one hour ahead, but
only 0.702 under the genuine 24-hour evaluation used throughout this
report. Roughly a fifth of its apparent skill turns out to be an
artefact of how the evaluation was set up, not a real property of the
model.

Permutation importance for the restricted 24-hour model shows that
hour-of-day position, the same-hour average over the previous seven
days, and the 24- and 168-hour lags dominate. When sensor and weather
features are added, accuracy falls rather than rises.

8. Foundation model

Chronos-Bolt (amazon/chronos-bolt-base) was applied zero-shot: no
gradient step was taken on this dataset, and the published weights were
used unchanged.

The model sees only past values of Appliances up to each forecast
origin. It is univariate, so sensors and weather are unavailable to it,
and it receives no timestamps, so any daily periodicity has to be
inferred from the numeric pattern alone.

It outputs quantiles directly, giving both a median point forecast and
prediction intervals. Inference for all 14 windows took 4.3 seconds on a
CPU.

Chronos-Bolt achieves MAE 33.63 Wh and MASE 0.630, the best point
accuracy of any individual model tested.

Its improvement over the weekly seasonal benchmark is 21.6% in MASE, and
unlike every other model in this report, that improvement is
statistically distinguishable from zero (−9.29 Wh, 95% interval
[−18.17, −1.51]).

The difference between Chronos and the target-only SARIMA, however, is
not resolvable (−3.16 Wh, [−9.43, +3.43]).

Chronos also reproduces the daily shape without being given the time of
day: the correlation between its mean forecast profile by hour and the
observed profile is 0.933. This supports reading its advantage as a
transfer of general temporal structure learned during pretraining rather
than as something specific to this household.

With only 168 hours of history, the model already reaches MASE 0.673 ---
better than a SARIMAX model chosen from a 147-model search over 17 weeks
of data. A new household could, in principle, be forecast to near-best
accuracy after just one week, with no training step at all.

At a common nominal 80% level, Chronos produces intervals with a mean
width of 91.0 Wh and 82.4% coverage, against 168.1 Wh at 90.2% coverage
for SARIMA and 154.4 Wh at 85.1% coverage for the benchmark. Its
intervals are roughly half the width while sitting closer to the nominal
level.

Two limitations should be flagged. Chronos-Bolt is trained on the
0.1--0.9 decile range only, so it cannot give reliable extreme-tail
quantiles without fine-tuning. Also, pretraining contamination cannot be
ruled out: the Appliances Energy Prediction dataset is a well-known
public benchmark, and any contamination would inflate exactly the one
clearly positive result reported here.

9. Results and error analysis

Model          Class             MAE (Wh)    RMSE (Wh)         MASE    Bias (Wh)

Chronos-Bolt   Foundation           33.63        66.01        0.630       −15.60
(zero-shot)

SARIMA         SARIMAX              36.79        64.21        0.689        −4.65
target-only

Boosted trees, Feature              37.48        61.08        0.702        +0.21
recursive

SARIMAX +      SARIMAX              37.90        64.26        0.710        −4.99
calendar

SARIMAX +      SARIMAX              38.37        64.74        0.719        −3.21
calendar +
weather

Boosted trees, Feature              38.56        65.76        0.723        −4.37
restricted

Boosted trees, Feature              38.59        65.27        0.723        −3.95
direct

Boosted        Feature              40.12        64.93        0.752        −2.02
trees + lagged
covariates

Weekly         Benchmark            42.92        79.67        0.804       −12.62
seasonal naive

Boosted        Conditional          45.98        68.21        0.862        +9.27
trees +
realised
covariates

Daily seasonal Benchmark            48.39        85.79        0.907        +1.67
naive

Mean           Benchmark            50.01        73.99        0.937        −3.06

Naive          Benchmark           143.38       170.62        2.687      +114.67

Drift          Benchmark           143.94       171.26        2.698      +115.31

The overall pattern is clear. Locally fitted SARIMA and boosted-tree
models improve point accuracy over the strongest benchmark, but their
advantage cannot be separated from it statistically in this small
sample. Of the thirteen models compared against the weekly seasonal
naive, only Chronos-Bolt is clearly better and only the naive and drift
forecasts are clearly worse; the remaining ten are statistically
indistinguishable from it.

Error is heavily concentrated in one test window. Window 8, 20 May 2016,
contributes 16.2% of total absolute error against the 7.1% expected
under an even spread across the 14 windows, and every model performs
poorly on it. This points to unusual household behaviour on that day
rather than a weakness specific to one method.

Horizon and time of day are also confounded here, since every forecast
origin falls at 17:00, so a given step always lands on the same clock
hour.

Chronos does not win on the easy overnight hours: at step 18, the
hardest point of the horizon, its MAE is 86.5 Wh against 113.1 Wh for
the benchmark and 90.3 Wh for SARIMA. Its advantage is earned precisely
where the errors are largest.

Forecast errors are highly correlated across models, with a mean
pairwise correlation of 0.839. A simple equal-weight combination of the
benchmark, SARIMA and boosted-tree forecasts achieves MASE 0.676 ---
better than any of its three members individually, and clearly better
than the benchmark alone (−6.86 Wh, [−13.46, −1.18]) --- though it
still falls short of Chronos.

Finally, Chronos carries the largest bias of any model at −15.60 Wh: it
tracks the daily shape well but systematically shaves off the peaks, the
expected behaviour of a model minimising a quantile loss on a
right-skewed series.

10. Discussion and limitations

The household series contains substantial recurring structure, but also
a large amount of variation that the available predictors simply do not
observe. That is why the seasonal benchmarks perform as well as they do,
and why ten of the thirteen models compared cluster together
indistinguishably.

Locally fitted complexity can refine the forecast a little, but the
additional information it extracts from sensors and weather is very
limited. This conclusion is reached independently through the partial
correlations in Section 3, the SARIMAX coefficients in Section 6, and
the feature importances and covariate variants in Section 7.

The foundation model tells a different story. Chronos-Bolt receives no
timestamps and no covariates, yet infers enough recurring structure to
beat the seasonal benchmark by a measurable margin, and it does so with
one week of history rather than seventeen. That advantage is consistent
with the transfer of general time-series structure learned elsewhere,
rather than with the extraction of extra household-specific information.

Limitations

The test sample contains only 14 rolling 24-hour windows from a
single household, which limits statistical power and how far the
conclusions generalise.

Pretraining contamination cannot be excluded for the foundation
model, and would inflate the one clearly positive result reported
here.

A single window contributes 16.2% of total error, so results would
shift noticeably if that day fell on the other side of the
train/test split.

All forecast origins occur at the same clock time, so horizon and
time of day are partially confounded throughout.

The target is strongly right-skewed and no log transform was
applied; this is the most obvious unexplored improvement.

Hyperparameters were selected on three expanding-window folds, and
cross-validation MAE predicted test-period accuracy only weakly. The
selection was left as made rather than revised on test performance,
but the reported figures carry that uncertainty.

Models were fitted once rather than refitted at each rolling origin.
This keeps the comparison like-for-like, but understates what an
operational system, refitted regularly, could achieve.

Future work should repeat the analysis across multiple households,
stagger forecast origins across the day rather than fixing them at
17:00, evaluate transformed-target models, refit at each origin, pursue
forecast combination more systematically, and investigate better methods
for extreme-tail quantiles.

11. Conclusion and recommendation

Household appliance energy use contains strong recurring temporal
structure, which is why a simple seasonal rule proves a surprisingly
strong baseline. Target-only SARIMA and the recursive boosted-tree model
both improved on it numerically, but the 14-window sample gives no
strong evidence that either improvement is reliable once bootstrap
uncertainty is accounted for, and adding weather and sensor covariates
made forecasts worse.

For this forecasting task, Chronos-Bolt is the preferred individual
model. It has the lowest MAE and MASE, is the only model whose
improvement over the strongest benchmark is statistically defensible,
needs no household-specific fitting, and produces the sharpest
probabilistic forecasts.

Its downsides are equally important: low interpretability, a substantial
software dependency, the largest negative bias of any model tested,
limited tail quantiles, and an open question about pretraining
contamination.

Criterion          Weekly         SARIMA         Boosted trees  Chronos-Bolt
seasonal naive

MASE               0.804          0.689 (n.s.)   0.702 (n.s.)   0.630
(significant)

Training required  None           147-model      CV + fit       None
search

Cold start         7 days         Weeks          Weeks          7 days, no
fitting

80% interval width 154 Wh         168 Wh         ---            91 Wh

Tail quantiles     Yes            Yes            No             No

Interpretability   Complete       Moderate       Low            Low

The weekly seasonal naive should be kept as the operational fallback. It
is transparent, almost free to run, easy to explain to a non-technical
stakeholder, and it pairs with the empirical hour-specific intervals of
Section 5, which were better shaped than SARIMA's Gaussian ones. Where
interpretability, minimal infrastructure or trustworthy tail uncertainty
matters more than a marginal gain in point accuracy, it remains
defensible on its own.

The broader conclusion is more interesting than the ranking. On a single
household with 17 weeks of history at a 24-hour horizon, locally fitted
complexity did not pay: a 147-model SARIMAX search and a tuned
gradient-boosting pipeline both landed within noise of a one-line
seasonal rule. What did pay was complexity learned elsewhere and
transferred in --- a finding that would have been invisible without an
evaluation protocol that measures what it claims to and confidence
intervals that show what fourteen days can resolve.

References

Alston, W.N. (2026) Forecasting Time Series in Python: Principles,
Models, and Modern Practice.

Ansari, A.F. et al. (2024) 'Chronos: learning the language of time
series', Transactions on Machine Learning Research.

Bates, J.M. and Granger, C.W.J. (1969) 'The combination of forecasts',
Operational Research Quarterly, 20(4), pp. 451--468.

Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017) 'Data driven
prediction models of energy use of appliances in a low-energy house',
Energy and Buildings, 140, pp. 81--97.

Chen, T. and Guestrin, C. (2016) 'XGBoost: a scalable tree boosting
system', Proceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining, pp. 785--794.

Hyndman, R.J. and Athanasopoulos, G. (2021) Forecasting: Principles and
Practice. 3rd edn. Melbourne: OTexts.

Hyndman, R.J. and Koehler, A.B. (2006) 'Another look at measures of
forecast accuracy', International Journal of Forecasting, 22(4),
pp. 679--688.

Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. and Shin, Y. (1992)
'Testing the null hypothesis of stationarity against the alternative of
a unit root', Journal of Econometrics, 54(1--3), pp. 159--178.

Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python',
Journal of Machine Learning Research, 12, pp. 2825--2830.
