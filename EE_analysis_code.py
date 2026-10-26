"""Employment Equity data: cleaning and analysis.

Run: python EE_analysis_code.py   (ee_meta.csv and ee_profile_all.csv in the same folder)
Steps: clean profile -> reshape meta wide-to-long -> merge -> flag dominant employer -> insights.
All comparisons use shares (%) and are checked with and without the single largest employer.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')   # no pop-up windows; charts are saved as PNG
import matplotlib.pyplot as plt

DATA_DIR = ''   # CSVs in the same folder as this script
OUT_DIR  = ''   # charts and tables are saved here

meta    = pd.read_csv(DATA_DIR + 'ee_meta.csv')
profile = pd.read_csv(DATA_DIR + 'ee_profile_all.csv')
print('meta:', meta.shape, '| profile:', profile.shape)

# 1. Clean the profile file
# Issues found: inconsistent column case, a `foriegn` typo, and the label `Temporary` (2024 only, 500 rows) vs `Temporary employees` (all other years).

profile.columns = profile.columns.str.lower().str.replace('foriegn', 'foreign')
profile['occupation_level'] = profile['occupation_level'].replace({'Temporary': 'Temporary employees'})

print(profile['occupation_level'].value_counts())            # 7 levels x 6000 rows
print('duplicate keys:', profile.duplicated(['submission_year','ee_ref','occupation_level']).sum())
print('negative counts:', (profile.select_dtypes('number').drop(columns='submission_year') < 0).sum().sum())

# 2. Reshape the meta file (one row per employer-year)
# The meta file has one row per employer and one column per attribute per year (73 columns).

stubs = ['dti_registration_name','industry_sector','seta_classification',
         'business_type','province_physical','city_town_physical']
meta_long = pd.wide_to_long(meta, stubnames=stubs, i='ee_ref',
                            j='submission_year', sep='_').reset_index()
for c in stubs:
    meta_long[c] = meta_long[c].str.strip().str.upper()
print(meta_long.shape)   # 6000 = 500 x 12

# 3. Merge

df = profile.merge(meta_long, on=['ee_ref','submission_year'], how='left', validate='m:1')
assert len(df) == len(profile) == 42000
race_gender = [c for c in df.columns if c.endswith(('_male','_female'))]
df['total'] = df[race_gender].sum(axis=1)
print(df.isnull().sum()[lambda s: s > 0])

# 4. One employer dominates the data
# A handful of extreme values (e.g. 73,872 African women in one cell) all belong to one employer, in every year. They are genuine, not typos.

big = df.groupby('ee_ref')['total'].sum().idxmax()
print(df.loc[df.ee_ref == big, 'dti_registration_name'].unique())

share_by_year = (df[df.ee_ref == big].groupby('submission_year')['total'].sum()
                 / df.groupby('submission_year')['total'].sum() * 100).round(1)
print('Share of all reported headcount (%):'); print(share_by_year)
print('Share over all years: %.1f%%' % (df.loc[df.ee_ref == big,'total'].sum() / df['total'].sum() * 100))

# 5. Long format (race x gender) for share calculations

long = df.melt(id_vars=['submission_year','ee_ref','occupation_level'],
               value_vars=race_gender, var_name='group', value_name='count')
long[['race','gender']] = long['group'].str.split('_', expand=True)
ex = long[long.ee_ref != big]      # excluding the dominant employer

# Insight A. Top management: race composition (robustness check included)

def race_share(d, level):
    t = d[d.occupation_level == level].groupby(['submission_year','race'])['count'].sum().unstack()
    return (t.div(t.sum(axis=1), axis=0) * 100).round(1)

top_ex  = race_share(ex,   'Top management')
top_all = race_share(long, 'Top management')
print(top_ex.loc[[2013, 2024]]); print(top_all.loc[[2013, 2024]])
print('Max abs. difference with vs without big employer (pp):', (top_all - top_ex).abs().max().max())
print('Top management headcount excl. big employer:', ex[ex.occupation_level=='Top management'].groupby('submission_year')['count'].sum().loc[[2013,2024]].to_dict())

colors = {'white':'#7b5ea7','african':'#1f77b4','indian':'#d62728','coloured':'#ff7f0e','foreign':'#2ca02c'}
fig, ax = plt.subplots(figsize=(10, 5.2))
for r in top_ex.columns:
    ax.plot(top_ex.index, top_ex[r], marker='o', lw=2, color=colors[r])
    ax.annotate(r.capitalize(), (2024, top_ex[r].iloc[-1]), xytext=(8, 0),
                textcoords='offset points', va='center', fontsize=10, color=colors[r])
ax.set_xlim(2012.7, 2026.4); ax.set_xticks(range(2013, 2025, 2))
ax.set_ylabel('% of top management'); ax.set_xlabel('Year')
ax.set_title('Top management: race share (excl. KZN Education)')
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT_DIR + 'top_mgmt_race.png', dpi=150); plt.close()
top_ex.to_csv(OUT_DIR + 'top_mgmt_race_share.csv')

# Insight B. Top management vs the whole workforce (2024)

wf = ex.groupby(['submission_year','race'])['count'].sum().unstack()
wf = (wf.div(wf.sum(axis=1), axis=0) * 100).round(1)
cmp = pd.DataFrame({'workforce_%': wf.loc[2024], 'top_mgmt_%': top_ex.loc[2024]})
print(cmp)

# Insight C. Women's share by occupational level

g = ex.groupby(['submission_year','occupation_level','gender'])['count'].sum().unstack()
g['women_%'] = (g['female'] / (g['female'] + g['male']) * 100).round(1)
w_full = g['women_%'].unstack('occupation_level')
w_tbl = w_full.loc[[2013, 2018, 2024]].T
print(w_tbl)

ov = ex.groupby(['submission_year','gender'])['count'].sum().unstack()
print('Women share of whole workforce (%):', (ov.female / (ov.female + ov.male) * 100).round(1).loc[[2013, 2024]].to_dict())

# chart: drop the volatile 'Temporary employees' series, highlight top and senior management
w_plot = w_full.rename(columns=lambda c: c.split(' and ')[0].split(',')[0]).drop(columns='Temporary employees')
offsets = {'Semi-skilled': 8, 'Skilled technical': 0, 'Professionally qualified': -8}
fig, ax = plt.subplots(figsize=(10, 5.2))
for c in w_plot.columns:
    hi = c in ['Top management', 'Senior management']
    ax.plot(w_plot.index, w_plot[c], lw=3 if hi else 1.5, alpha=1 if hi else 0.5, marker='o' if hi else None)
    ax.annotate(c, (2024, w_plot[c].iloc[-1]), xytext=(6, offsets.get(c, 0)),
                textcoords='offset points', fontsize=9, va='center')
ax.set_xlim(2013, 2027.5); ax.set_xticks(range(2013, 2025, 2))
ax.set_ylabel('% women'); ax.set_xlabel('Year')
ax.set_title('Women share by occupational level (excl. KZN Education)')
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT_DIR + 'women_share.png', dpi=150); plt.close()
w_tbl.to_csv(OUT_DIR + 'women_share_by_level.csv')

# Insight D. Data quality: classification labels change over time for the same employer

chk = df.groupby('ee_ref')[['industry_sector','business_type','province_physical']].nunique()
print('Employers whose label changes across years:'); print((chk > 1).sum())
print('As % of employers:'); print(((chk > 1).mean() * 100).round(1))
print('Missing registration name (pct of employer-years):', round(df['dti_registration_name'].isna().mean() * 100, 1))