const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, ImageRun,
} = require("docx");

const NAVY = "1F3864";
const GREY = "595959";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20, color: GREY })],
    spacing: { after: 240 },
    alignment: AlignmentType.CENTER,
  });
}
function img(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height } })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
  });
}

function makeTable(headerRow, rows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  const headerCells = headerRow.map((t, i) => new TableCell({
    width: { size: colWidths[i], type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: NAVY },
    children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: "FFFFFF", size: 20 })] })],
  }));
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((t, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ri % 2 === 0 ? "F2F2F2" : "FFFFFF" },
      children: [new Paragraph({ children: [new TextRun({ text: String(t), size: 20 })] })],
    })),
  }));
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [new TableRow({ children: headerCells }), ...bodyRows],
  });
}

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22 } },
    },
  },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Indian Two-Wheeler Market Assessment", bold: true, size: 44, color: NAVY })],
        spacing: { after: 80 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Capstone Assignment 20.1 — Initial Report & Exploratory Data Analysis", size: 26, color: GREY })],
        spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Prepared: July 2026", size: 22, color: GREY, italics: true })],
        spacing: { after: 300 },
      }),

      h1("1. Summary of Findings"),
      new Paragraph({
        children: [
          new TextRun({ text: "This report summarizes the exploratory data analysis (EDA), data cleaning, feature engineering, and baseline modeling performed for the capstone research question: which factors most significantly predict two-wheeler sales volume and resale value across Indian states, and can a machine-learning model reliably forecast regional demand by brand and model segment? The full analysis, code, and chart outputs are in the accompanying Jupyter notebook, ", size: 22 }),
          new TextRun({ text: "Indian_Two_Wheeler_EDA.ipynb", bold: true, size: 22 }),
          new TextRun({ text: ", submitted alongside this document.", size: 22 }),
        ],
        spacing: { after: 200 },
      }),

      new Paragraph({
        children: [
          new TextRun({ text: "Data source: ", bold: true, size: 22 }),
          new TextRun({
            text: "This analysis uses the user-provided dataset bike_sales_india.csv — 10,000 two-wheeler sale records across 10 states, 8 brands, and 40 models, with State, Average Daily Distance, Brand, Model, Price, Year of Manufacture, Engine Capacity, Fuel Type, Mileage, Owner Type, Registration Year, Insurance Status, Seller Type, Resale Price, and City Tier. The dataset arrived with no missing values and no duplicate rows; the cleaning steps below (deduplication, imputation, outlier capping, consistency checks) are still applied in full as a defensive, reusable pipeline.",
            size: 20, color: GREY,
          }),
        ],
        spacing: { after: 240 },
        shading: { type: ShadingType.CLEAR, fill: "E9F0FB" },
      }),

      h2("Headline results"),
      bullet("10,000 records across 10 states, 8 brands, and 4 engineered vehicle segments (Commuter, Sports, Cruiser, Superbike) plus Electric, after a full cleaning and validation pass (0 duplicates, 0 missing values, all resale-price and registration-year consistency checks passed)."),
      bullet("Original price and vehicle age are the dominant predictors of resale price; the Linear Regression baseline already explains ~81% of variance (R\u00b2 = 0.809), and ensemble models (Bagging, Random Forest, Gradient Boosting) perform in the same range rather than decisively beating it."),
      bullet("That near-tie is itself a finding: it indicates resale price in this dataset follows a fairly linear price/age relationship, with limited additional non-linear brand \u00d7 segment interaction for tree-based models to exploit \u2014 unlike what might be expected from real-world brand-specific depreciation curves."),
      bullet("For the demand classification task, ensemble classifiers (Gradient Boosting, Random Forest) modestly outperform logistic regression (F1 = 0.810 vs. 0.788 baseline; ROC-AUC = 0.861 vs. 0.849), suggesting more non-linear structure in demand-tier prediction than in resale pricing."),
      bullet("K-Means clustering of the 10 states on price, usage, resale ratio, EV share, and vehicle age surfaces distinguishable regional buyer profiles for inventory planning."),

      h1("2. Methodology"),
      h2("2.1 Data Cleaning"),
      bullet("Standardized column names to snake_case and text fields (trimmed whitespace, normalized casing across state, brand, model, fuel type, owner type, insurance status, seller type, city tier)."),
      bullet("Checked for and removed exact duplicate records (0 found)."),
      bullet("Audited and imputed missing values \u2014 median for numeric fields, mode for categorical fields (0 missing values found in this file; imputation logic retained for pipeline robustness)."),
      bullet("Validated logical consistency: no records had resale price exceeding original price, and no records had a registration year preceding the manufacture year."),
      bullet("Detected and capped outliers in price, resale price, average daily distance, and mileage using the IQR (1.5\u00d7) method."),

      h2("2.2 Feature Engineering"),
      bullet("vehicle_age_years \u2014 derived from registration year."),
      bullet("resale_ratio \u2014 resale price as a fraction of original price; the core depreciation benchmark requested in the proposal."),
      bullet("segment \u2014 Commuter / Sports / Cruiser / Superbike, inferred from engine capacity since the raw file has no explicit body-style column; Electric vehicles are grouped as their own segment."),
      bullet("usage_band / price_band \u2014 categorical buckets (Low/Medium/High daily distance; Economy/Mid/Premium price via quantile cuts) for segment profiling."),
      bullet("is_electric \u2014 binary powertrain flag."),
      bullet("high_demand \u2014 classification label: 1 if a (state, brand, segment) combination sells above the median volume for that state, else 0."),

      h2("2.3 Modeling"),
      bullet("Regression \u2014 predict resale_price. Baseline: Linear Regression. Ensembles: Bagging Regressor, Random Forest Regressor, Gradient Boosting Regressor."),
      bullet("Classification \u2014 predict high_demand. Baseline: Logistic Regression. Comparisons: Decision Tree, Random Forest, AdaBoost, Gradient Boosting."),
      bullet("Clustering \u2014 K-Means (k=3) on state-level aggregated buyer-behavior features to group states by demand pattern."),

      h1("3. Exploratory Data Analysis \u2014 Selected Visuals"),

      h2("3.1 Brand Market Share"),
      img("report_images/fig_02_cell20.png", 5800000 / 9525, 3200000 / 9525),
      caption("Figure 1. Brand market share and sales volume across the sample \u2014 the 8 brands are close to evenly represented."),

      h2("3.2 Resale Ratio by Vehicle Age"),
      img("report_images/fig_04_cell24.png", 5000000 / 9525, 3400000 / 9525),
      caption("Figure 2. Resale ratio vs. vehicle age by segment."),

      h2("3.3 Average Resale Price by State"),
      img("report_images/fig_07_cell30.png", 5500000 / 9525, 3400000 / 9525),
      caption("Figure 3. Regional variation in average resale price \u2014 a proxy for the 'regional demand map' deliverable in the proposal."),

      h1("4. Model Performance"),
      h2("4.1 Resale Price Regression"),
      makeTable(
        ["Model", "RMSE (INR)", "MAE (INR)", "R\u00b2"],
        [
          ["Linear Regression (baseline)", "28,929", "22,702", "0.809"],
          ["Gradient Boosting Regressor", "28,974", "22,723", "0.809"],
          ["Random Forest Regressor", "29,363", "22,963", "0.804"],
          ["Bagging Regressor", "29,616", "23,104", "0.800"],
        ],
        [3600, 1800, 1800, 1300],
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      img("report_images/fig_08_cell36.png", 5000000 / 9525, 3000000 / 9525),
      caption("Figure 4. RMSE comparison \u2014 the linear baseline is competitive with, or marginally ahead of, the ensemble regressors on this dataset."),

      new Paragraph({
        children: [new TextRun({
          text: "Evaluation metric rationale: RMSE is reported in INR so the error is directly interpretable by dealers and manufacturers; MAE is included because it is less sensitive to residual outliers in resale price; R\u00b2 communicates the share of variance in resale price explained by each model, giving a normalized basis for comparing model iterations. Unlike a typical capstone dataset where ensembles clearly beat a linear baseline, here the four models land within roughly 2% of each other on RMSE \u2014 a signal that resale price in this data is close to a linear function of price and age, with little extra non-linear signal for trees to capture.",
          size: 22,
        })],
        spacing: { after: 200 },
      }),

      h2("4.2 High vs. Low Demand Classification"),
      makeTable(
        ["Model", "Accuracy", "F1-score", "ROC-AUC"],
        [
          ["Logistic Regression (baseline)", "0.738", "0.788", "0.849"],
          ["Decision Tree", "0.743", "0.781", "0.857"],
          ["Random Forest", "0.746", "0.807", "0.854"],
          ["AdaBoost", "0.753", "0.786", "0.859"],
          ["Gradient Boosting", "0.752", "0.810", "0.861"],
        ],
        [3600, 1600, 1600, 1700],
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      new Paragraph({
        children: [new TextRun({
          text: "Evaluation metric rationale: F1-score is the primary metric because it balances the cost of false 'high-demand' calls (overstock risk for a dealer) against false 'low-demand' calls (lost-sale risk); ROC-AUC evaluates ranking quality independent of the classification threshold; accuracy is reported as an intuitive sanity check. Note: state, brand, and segment identity were deliberately excluded from the classifier's input features because the demand label is itself derived from those columns \u2014 including them would let the model memorize the label rather than learn a generalizable relationship. Gradient Boosting gives the best F1 and ROC-AUC, indicating a modest but real non-linear signal in how vehicle/customer attributes (price, mileage, usage, ownership history) relate to regional demand tier.",
          size: 22,
        })],
        spacing: { after: 200 },
      }),

      h2("4.3 State Clustering"),
      img("report_images/fig_10_cell42.png", 5200000 / 9525, 3400000 / 9525),
      caption("Figure 5. K-Means clusters of the 10 states by average price, EV share, and usage intensity \u2014 a first pass at the 'segment profiles by geography' deliverable."),

      h1("5. Key Takeaways"),
      bullet("Original price and vehicle age are the dominant predictors of resale price, and the relationship is close to linear in this dataset \u2014 ensembles do not decisively outperform the linear baseline, which is itself a useful diagnostic about the data-generating process behind this file."),
      bullet("Demand classification shows more non-linear structure than resale pricing: ensemble classifiers (Gradient Boosting, Random Forest) provide a modest, consistent edge over logistic regression."),
      bullet("Regional variation in resale price exists across the 10 states but is comparatively narrow, suggesting demand intensity is a more differentiating regional signal than raw resale value for inventory planning."),
      bullet("State clustering identifies distinguishable buyer-behavior groups, supporting differentiated inventory strategy by region rather than a single national playbook."),
      bullet("Data-quality checks (duplicates, missing values, price/date consistency) all passed cleanly, so the modeling results reflect the underlying signal in the data rather than cleaning artifacts."),

      h1("6. Next Steps"),
      bullet("Hyperparameter tuning (grid/randomized search) on the leading models for both tasks (Gradient Boosting)."),
      bullet("Incorporate a genuine transaction-date field if one becomes available, to model demand trends over time rather than a single-snapshot demand label."),
      bullet("Test whether model-level (not just brand-level) features sharpen resale-price predictions, and whether additional real-world fields (e.g., accident history, service records) would strengthen the largely-linear price/age relationship found here."),
      bullet("Revisit the engineered segment labels (currently inferred from engine capacity) against any authoritative body-style classification if one becomes available, since the current EDA showed engine capacity is close to independently distributed across brands in this file."),

      new Paragraph({ text: "", spacing: { after: 200 } }),
      new Paragraph({
        children: [new TextRun({ text: "Jupyter Notebook: ", bold: true, size: 22 }), new TextRun({ text: "Indian_Two_Wheeler_EDA.ipynb (submitted alongside this report)", size: 22 })],
      }),
      new Paragraph({
        children: [new TextRun({ text: "Supporting files: ", bold: true, size: 22 }), new TextRun({ text: "bike_sales_india.csv (dataset used)", size: 22 })],
      }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Indian_Two_Wheeler_Capstone_Report.docx", buf);
  console.log("Report written");
});
