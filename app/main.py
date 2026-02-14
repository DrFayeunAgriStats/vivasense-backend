import { useState, useRef, useCallback } from "react";
import { Layout } from "@/components/layout/Layout";
import { VivaSenseHero } from "@/components/vivasense/VivaSenseHero";
import { VivaSenseFeatures } from "@/components/vivasense/VivaSenseFeatures";
import { VivaSenseInstitutional } from "@/components/vivasense/VivaSenseInstitutional";
import { useToast } from "@/hooks/use-toast";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Loader2, Upload, Download, RefreshCw, FileDown, Info } from "lucide-react";

const VIVASENSE_API_URL = "https://vivasense-backend.onrender.com/vivasense/run";

interface ANOVARow {
  Source: string;
  SS: number;
  DF: number;
  F: number;
  p_value: number;
}

interface GroupMean {
  Treatment?: string;
  Mean: number;
  SD?: number;
  N?: number;
}

interface TukeyComparison {
  group1: string;
  group2: string;
  meandiff?: number;
  diff?: number;
  p_value?: number;
  pValue?: number;
  reject?: boolean;
  significant?: boolean;
}

interface Assumption {
  statistic: number | null;
  p_value: number | null;
  pass: boolean | null;
}

interface VivaSenseResults {
  anova_table: ANOVARow[];
  group_means: GroupMean[];
  tukey_hsd?: TukeyComparison[];
  assumptions?: {
    shapiro_wilk: Assumption;
    levene_test: Assumption;
  };
  shapiro_wilk?: Assumption;
  levene_test?: Assumption;
  interpretation: string;
  metadata?: {
    n_observations: number;
    n_groups: number;
    outcome_variable: string;
    predictor_variables: string[];
  };
}

export default function VivaSense() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<VivaSenseResults | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [outcome, setOutcome] = useState("");
  const [predictors, setPredictors] = useState("");
  const [error, setError] = useState<string | null>(null);
  const formRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToForm = useCallback(() => {
    formRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError("Please upload a dataset file");
      return;
    }
    if (!outcome) {
      setError("Please enter the outcome variable name");
      return;
    }
    if (!predictors) {
      setError("Please enter the predictor variable name(s)");
      return;
    }

    setIsLoading(true);
    setResults(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("outcome", outcome);
    formData.append("predictors", predictors);

    try {
      const response = await fetch(VIVASENSE_API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMsg = `Analysis failed (${response.status})`;
        try {
          const errorData = JSON.parse(errorText);
          if (errorData.detail) {
            errorMsg = typeof errorData.detail === 'string' 
              ? errorData.detail 
              : JSON.stringify(errorData.detail);
          }
        } catch {
          errorMsg = errorText || errorMsg;
        }
        throw new Error(errorMsg);
      }

      const data: VivaSenseResults = await response.json();
      setResults(data);
      
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 100);

      toast({
        title: "Analysis Complete",
        description: "Your results are ready below.",
      });
    } catch (error) {
      console.error("VivaSense analysis error:", error);
      const errMsg = error instanceof Error ? error.message : "An unexpected error occurred. Please try again.";
      setError(errMsg);
      toast({
        title: "Analysis Failed",
        description: errMsg,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearResults = () => {
    setResults(null);
    setFile(null);
    setOutcome("");
    setPredictors("");
    setError(null);
  };

  // ========== DOWNLOAD FUNCTIONS ==========
  const downloadCSV = (data: any[], filename: string) => {
    if (!data || data.length === 0) {
      toast({
        title: "No Data",
        description: "No data available to download",
        variant: "destructive",
      });
      return;
    }
    
    // Get headers from first object
    const headers = Object.keys(data[0]);
    
    // Create CSV content
    const csvContent = [
      headers.join(','),
      ...data.map(row => 
        headers.map(header => {
          const value = row[header];
          // Handle different types and escape commas
          if (value === null || value === undefined) return '';
          const strValue = String(value);
          return strValue.includes(',') ? `"${strValue}"` : strValue;
        }).join(',')
      )
    ].join('\n');
    
    // Create and trigger download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Downloaded",
      description: `${filename} saved successfully`,
    });
  };

  const downloadANOVA = () => {
    if (results?.anova_table) {
      downloadCSV(results.anova_table, 'VivaSense_ANOVA_Table.csv');
    }
  };

  const downloadGroupMeans = () => {
    if (results?.group_means) {
      downloadCSV(results.group_means, 'VivaSense_Group_Means.csv');
    }
  };

  const downloadTukey = () => {
    if (results?.tukey_hsd) {
      downloadCSV(results.tukey_hsd, 'VivaSense_Tukey_HSD.csv');
    }
  };

  // ========== PLAIN ENGLISH INTERPRETATION ==========
  const generatePlainEnglishInterpretation = () => {
    if (!results) return "";
    
    // Find treatment row in ANOVA
    const treatmentRow = results.anova_table.find(row => 
      row.Source?.toLowerCase().includes('treatment') || 
      row.Source?.toLowerCase() === predictors.toLowerCase().trim()
    );
    
    if (!treatmentRow) return results.interpretation;
    
    const fValue = treatmentRow.F?.toFixed(2) || "N/A";
    const pValue = treatmentRow.p_value;
    const pDisplay = pValue < 0.001 ? "< 0.001" : pValue?.toFixed(4) || "N/A";
    
    // Analyze group means
    const means = results.group_means || [];
    const sortedMeans = [...means].sort((a, b) => b.Mean - a.Mean);
    const highest = sortedMeans[0];
    const lowest = sortedMeans[sortedMeans.length - 1];
    
    const highestName = Object.values(highest)[0];
    const lowestName = Object.values(lowest)[0];
    const highestMean = highest.Mean?.toFixed(1);
    const lowestMean = lowest.Mean?.toFixed(1);
    const percentDiff = ((highest.Mean - lowest.Mean) / lowest.Mean * 100).toFixed(1);
    
    // Check assumptions
    const shapiro = results.assumptions?.shapiro_wilk || results.shapiro_wilk;
    const levene = results.assumptions?.levene_test || results.levene_test;
    const assumptionsPass = (shapiro?.pass && levene?.pass);
    
    // Build interpretation
    let text = "";
    
    // Main finding
    if (pValue !== null && pValue < 0.05) {
      text = `The ANOVA results show that **${predictors} had a significant effect** on ${outcome} (F = ${fValue}, p ${pDisplay}).\n\n`;
      
      text += `**Key Findings:**\n`;
      text += `• ${highestName} produced the highest mean value (${highestMean})\n`;
      text += `• ${lowestName} had the lowest mean value (${lowestMean})\n`;
      text += `• This represents a **${percentDiff}% difference** between highest and lowest treatments\n\n`;
      
      // Tukey results
      if (results.tukey_hsd && results.tukey_hsd.length > 0) {
        const sigComparisons = results.tukey_hsd.filter(c => {
          const p = c.p_value ?? c.pValue;
          return c.reject ?? c.significant ?? (p !== undefined && p < 0.05);
        });
        text += `**Post-hoc Analysis (Tukey HSD):**\n`;
        text += `• ${sigComparisons.length} out of ${results.tukey_hsd.length} pairwise comparisons were statistically significant\n`;
        
        if (sigComparisons.length > 0) {
          text += `• Significant differences found between:\n`;
          sigComparisons.slice(0, 3).forEach(c => {
            const diff = (c.meandiff ?? c.diff)?.toFixed(1);
            text += `  - ${c.group1} vs ${c.group2} (difference: ${diff})\n`;
          });
          if (sigComparisons.length > 3) {
            text += `  - ...and ${sigComparisons.length - 3} more\n`;
          }
        }
      }
    } else {
      text = `The ANOVA results show **no significant effect** of ${predictors} on ${outcome} (F = ${fValue}, p = ${pDisplay}).\n\n`;
      
      text += `**Interpretation:**\n`;
      text += `• While ${highestName} had the numerically highest mean (${highestMean})\n`;
      text += `• And ${lowestName} had the lowest mean (${lowestMean})\n`;
      text += `• The ${percentDiff}% difference could be due to **random variation**\n`;
      text += `• No statistically significant differences were detected between treatments\n\n`;
    }
    
    // Assumptions
    text += `\n**Statistical Assumptions:**\n`;
    if (assumptionsPass) {
      text += `✓ Normality test (Shapiro-Wilk): PASS (p = ${shapiro?.p_value?.toFixed(3)})\n`;
      text += `✓ Equal variance test (Levene): PASS (p = ${levene?.p_value?.toFixed(3)})\n`;
      text += `✓ The ANOVA results are valid and reliable.\n`;
    } else {
      if (!shapiro?.pass) {
        text += `⚠️ Normality test (Shapiro-Wilk): FAIL (p = ${shapiro?.p_value?.toFixed(3)})\n`;
        text += `  → Data may not be normally distributed\n`;
      } else {
        text += `✓ Normality test: PASS\n`;
      }
      
      if (!levene?.pass) {
        text += `⚠️ Equal variance test (Levene): FAIL (p = ${levene?.p_value?.toFixed(3)})\n`;
        text += `  → Variances across groups may be unequal\n`;
      } else {
        text += `✓ Equal variance test: PASS\n`;
      }
      
      text += `\n**Recommendation:** Consider data transformation (log, square root) or non-parametric alternatives (Kruskal-Wallis test).\n`;
    }
    
    // Sample size note
    if (results.metadata) {
      text += `\n**Sample Size:** ${results.metadata.n_observations} observations across ${results.metadata.n_groups} groups\n`;
    }
    
    return text;
  };

  // ========== GENERATE LETTER GROUPS FROM TUKEY ==========
  const generateLetterGroups = () => {
    if (!results?.group_means || !results?.tukey_hsd) return [];
    
    // Get all treatment names and their means
    const treatments = results.group_means.map(row => ({
      name: String(Object.values(row)[0]),
      mean: row.Mean
    }));
    
    // Sort by mean (highest to lowest)
    treatments.sort((a, b) => b.mean - a.mean);
    
    // Build a matrix of which treatments are significantly different
    const sigDifferent: {[key: string]: Set<string>} = {};
    treatments.forEach(t => {
      sigDifferent[t.name] = new Set<string>();
    });
    
    // Populate from Tukey results
    results.tukey_hsd.forEach(comparison => {
      const pVal = comparison.p_value ?? comparison.pValue;
      const isSig = comparison.reject ?? comparison.significant ?? (pVal !== undefined && pVal < 0.05);
      
      if (isSig) {
        sigDifferent[comparison.group1]?.add(comparison.group2);
        sigDifferent[comparison.group2]?.add(comparison.group1);
      }
    });
    
    // Assign letters using compact letter display algorithm
    const letters: {[key: string]: string[]} = {};
    const alphabet = 'abcdefghijklmnopqrstuvwxyz'.split('');
    let currentLetter = 0;
    
    treatments.forEach((treatment, idx) => {
      letters[treatment.name] = [];
      
      if (idx === 0) {
        // First (highest) treatment gets 'a'
        letters[treatment.name].push(alphabet[currentLetter]);
      } else {
        // Check which treatments this one is NOT significantly different from
        let needsNewLetter = true;
        
        for (let i = 0; i < idx; i++) {
          const prevTreatment = treatments[i];
          const isDifferent = sigDifferent[treatment.name]?.has(prevTreatment.name);
          
          if (!isDifferent) {
            // Not significantly different - can share letters
            letters[treatment.name].push(...letters[prevTreatment.name]);
            needsNewLetter = false;
          }
        }
        
        if (needsNewLetter || letters[treatment.name].length === 0) {
          // Significantly different from all previous - needs new letter
          currentLetter++;
          letters[treatment.name].push(alphabet[currentLetter]);
        }
      }
      
      // Remove duplicates and sort
      letters[treatment.name] = [...new Set(letters[treatment.name])].sort();
    });
    
    // Format output
    return treatments.map(t => ({
      Treatment: t.name,
      Mean: t.mean.toFixed(2),
      Letters: letters[t.name].join('')
    }));
  };

  return (
    <Layout>
      <VivaSenseHero onGetStarted={scrollToForm} />
      <VivaSenseFeatures />

      {/* Analysis Form */}
      <section ref={formRef} className="py-16 bg-gradient-to-b from-white to-gray-50">
        <div className="container mx-auto px-4 max-w-3xl">
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Run Your Analysis</CardTitle>
              <CardDescription>
                Upload your dataset and specify your variables to get started
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* File Upload */}
                <div className="space-y-2">
                  <Label htmlFor="file">Dataset File (Excel or CSV)</Label>
                  <div className="flex items-center gap-4">
                    <Input
                      id="file"
                      type="file"
                      accept=".xlsx,.xls,.csv"
                      onChange={handleFileChange}
                      disabled={isLoading}
                      className="flex-1"
                    />
                    {file && (
                      <span className="text-sm text-green-600 flex items-center gap-1">
                        <Upload className="h-4 w-4" />
                        {file.name}
                      </span>
                    )}
                  </div>
                </div>

                {/* Outcome Variable */}
                <div className="space-y-2">
                  <Label htmlFor="outcome">Outcome Variable</Label>
                  <Input
                    id="outcome"
                    type="text"
                    placeholder="e.g., Yield_kg_ha"
                    value={outcome}
                    onChange={(e) => setOutcome(e.target.value)}
                    disabled={isLoading}
                  />
                  <p className="text-sm text-gray-500">
                    The column name of your response variable (must match exactly as in your file)
                  </p>
                </div>

                {/* Predictors */}
                <div className="space-y-2">
                  <Label htmlFor="predictors">Predictor Variable(s)</Label>
                  <Input
                    id="predictors"
                    type="text"
                    placeholder="e.g., Treatment"
                    value={predictors}
                    onChange={(e) => setPredictors(e.target.value)}
                    disabled={isLoading}
                  />
                  <p className="text-sm text-gray-500">
                    Treatment or factor column name (for multiple, separate with commas)
                  </p>
                </div>

                {/* Error Alert */}
                {error && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {/* Submit Button */}
                <Button
                  type="submit"
                  disabled={isLoading}
                  className="w-full"
                  size="lg"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing... (may take up to 60 seconds on first run)
                    </>
                  ) : (
                    "Analyze Data"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Results Section */}
      {results && (
        <section ref={resultsRef} className="py-16 bg-white">
          <div className="container mx-auto px-4 max-w-5xl">
            <div className="flex justify-between items-center mb-8">
              <h2 className="text-3xl font-bold">Analysis Results</h2>
              <Button onClick={handleClearResults} variant="outline">
                <RefreshCw className="mr-2 h-4 w-4" />
                New Analysis
              </Button>
            </div>

            {/* Metadata */}
            {results.metadata && (
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Dataset Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="font-semibold">Observations:</span> {results.metadata.n_observations}
                    </div>
                    <div>
                      <span className="font-semibold">Groups:</span> {results.metadata.n_groups}
                    </div>
                    <div>
                      <span className="font-semibold">Outcome:</span> {results.metadata.outcome_variable}
                    </div>
                    <div>
                      <span className="font-semibold">Predictors:</span> {results.metadata.predictor_variables.join(", ")}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* ========== PLAIN ENGLISH INTERPRETATION ========== */}
            <Alert className="mb-6 border-l-4 border-l-blue-600 bg-blue-50">
              <Info className="h-5 w-5 text-blue-600" />
              <AlertTitle className="text-lg font-semibold text-blue-900 mb-2">
                Plain-English Interpretation
              </AlertTitle>
              <AlertDescription className="text-gray-700 whitespace-pre-line leading-relaxed">
                {generatePlainEnglishInterpretation()}
              </AlertDescription>
            </Alert>

            {/* ANOVA Table with Download */}
            <Card className="mb-6">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-lg">ANOVA Table</CardTitle>
                  <CardDescription>Analysis of Variance Results</CardDescription>
                </div>
                <Button onClick={downloadANOVA} variant="outline" size="sm">
                  <FileDown className="mr-2 h-4 w-4" />
                  Download CSV
                </Button>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left font-semibold">Source</th>
                        <th className="px-4 py-2 text-right font-semibold">SS</th>
                        <th className="px-4 py-2 text-right font-semibold">DF</th>
                        <th className="px-4 py-2 text-right font-semibold">F</th>
                        <th className="px-4 py-2 text-right font-semibold">p-value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.anova_table.map((row, i) => (
                        <tr key={i} className="border-t">
                          <td className="px-4 py-2 font-medium">{row.Source}</td>
                          <td className="px-4 py-2 text-right">{row.SS?.toFixed(2) ?? 'N/A'}</td>
                          <td className="px-4 py-2 text-right">{row.DF ?? 'N/A'}</td>
                          <td className="px-4 py-2 text-right">{row.F?.toFixed(4) ?? 'N/A'}</td>
                          <td className="px-4 py-2 text-right font-semibold">
                            {row.p_value !== null && row.p_value !== undefined
                              ? row.p_value < 0.001
                                ? '<0.001'
                                : row.p_value.toFixed(4)
                              : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Group Means with Download */}
            <Card className="mb-6">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-lg">Group Means</CardTitle>
                  <CardDescription>Treatment group statistics</CardDescription>
                </div>
                <Button onClick={downloadGroupMeans} variant="outline" size="sm">
                  <FileDown className="mr-2 h-4 w-4" />
                  Download CSV
                </Button>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left font-semibold">Treatment</th>
                        <th className="px-4 py-2 text-right font-semibold">Mean</th>
                        {results.group_means[0]?.SD !== undefined && (
                          <th className="px-4 py-2 text-right font-semibold">SD</th>
                        )}
                        {results.group_means[0]?.N !== undefined && (
                          <th className="px-4 py-2 text-right font-semibold">N</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {results.group_means.map((row, i) => (
                        <tr key={i} className="border-t">
                          <td className="px-4 py-2 font-medium">{Object.values(row)[0]}</td>
                          <td className="px-4 py-2 text-right font-semibold">{row.Mean.toFixed(2)}</td>
                          {row.SD !== undefined && (
                            <td className="px-4 py-2 text-right">{row.SD.toFixed(2)}</td>
                          )}
                          {row.N !== undefined && (
                            <td className="px-4 py-2 text-right">{row.N}</td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Mean Separation with Letters */}
            {results.tukey_hsd && results.tukey_hsd.length > 0 && (
              <Card className="mb-6 border-l-4 border-l-green-500">
                <CardHeader className="flex flex-row items-center justify-between">
                  <div>
                    <CardTitle className="text-lg">Mean Separation (Letter Grouping)</CardTitle>
                    <CardDescription>Treatments with the same letter are not significantly different</CardDescription>
                  </div>
                  <Button 
                    onClick={() => {
                      const letterData = generateLetterGroups();
                      downloadCSV(letterData, 'VivaSense_Mean_Separation.csv');
                    }} 
                    variant="outline" 
                    size="sm"
                  >
                    <FileDown className="mr-2 h-4 w-4" />
                    Download CSV
                  </Button>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left font-semibold">Rank</th>
                          <th className="px-4 py-2 text-left font-semibold">Treatment</th>
                          <th className="px-4 py-2 text-right font-semibold">Mean</th>
                          <th className="px-4 py-2 text-center font-semibold">Letter Group</th>
                        </tr>
                      </thead>
                      <tbody>
                        {generateLetterGroups().map((row, i) => (
                          <tr key={i} className="border-t">
                            <td className="px-4 py-2 text-gray-600">{i + 1}</td>
                            <td className="px-4 py-2 font-medium">{row.Treatment}</td>
                            <td className="px-4 py-2 text-right font-semibold">{row.Mean}</td>
                            <td className="px-4 py-2 text-center">
                              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-bold bg-green-100 text-green-800">
                                {row.Letters}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-4 p-3 bg-blue-50 rounded-lg text-sm text-gray-700">
                    <strong>How to read:</strong> Treatments sharing at least one letter are not significantly different at α = 0.05. 
                    For example, if Treatment A has letters "a" and Treatment B has "ab", they are not significantly different.
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Tukey HSD with Download */}
            {results.tukey_hsd && results.tukey_hsd.length > 0 && (
              <Card className="mb-6">
                <CardHeader className="flex flex-row items-center justify-between">
                  <div>
                    <CardTitle className="text-lg">Post-Hoc Test (Tukey HSD)</CardTitle>
                    <CardDescription>Pairwise comparisons between treatment groups</CardDescription>
                  </div>
                  <Button onClick={downloadTukey} variant="outline" size="sm">
                    <FileDown className="mr-2 h-4 w-4" />
                    Download CSV
                  </Button>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left font-semibold">Comparison</th>
                          <th className="px-4 py-2 text-right font-semibold">Mean Diff</th>
                          <th className="px-4 py-2 text-right font-semibold">p-value</th>
                          <th className="px-4 py-2 text-center font-semibold">Significant?</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.tukey_hsd.map((row, i) => {
                          const pValue = row.p_value ?? row.pValue;
                          const diff = row.meandiff ?? row.diff;
                          const sig = row.reject ?? row.significant ?? (pValue !== undefined && pValue < 0.05);
                          
                          return (
                            <tr key={i} className="border-t">
                              <td className="px-4 py-2 font-medium">{row.group1} vs {row.group2}</td>
                              <td className="px-4 py-2 text-right">{diff?.toFixed(2) ?? 'N/A'}</td>
                              <td className="px-4 py-2 text-right">
                                {pValue !== null && pValue !== undefined
                                  ? pValue < 0.001
                                    ? '<0.001'
                                    : pValue.toFixed(4)
                                  : 'N/A'}
                              </td>
                              <td className="px-4 py-2 text-center">
                                {sig ? (
                                  <span className="text-red-600 font-semibold">Yes</span>
                                ) : (
                                  <span className="text-gray-500">No</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Assumptions */}
            {(results.assumptions || results.shapiro_wilk || results.levene_test) && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Assumption Tests</CardTitle>
                  <CardDescription>Statistical tests for ANOVA validity</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Shapiro-Wilk */}
                  {(() => {
                    const sw = results.assumptions?.shapiro_wilk || results.shapiro_wilk;
                    if (!sw) return null;
                    return (
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <h4 className="font-semibold mb-2">Shapiro-Wilk Test (Normality)</h4>
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600">Statistic:</span> {sw.statistic?.toFixed(4) ?? 'N/A'}
                          </div>
                          <div>
                            <span className="text-gray-600">p-value:</span> {sw.p_value?.toFixed(4) ?? 'N/A'}
                          </div>
                          <div>
                            <span className="text-gray-600">Result:</span>{' '}
                            <span className={sw.pass ? 'text-green-600 font-semibold' : 'text-red-600 font-semibold'}>
                              {sw.pass ? 'PASS ✓' : 'FAIL ⚠'}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })()}

                  {/* Levene Test */}
                  {(() => {
                    const lv = results.assumptions?.levene_test || results.levene_test;
                    if (!lv) return null;
                    return (
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <h4 className="font-semibold mb-2">Levene Test (Homogeneity of Variance)</h4>
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600">Statistic:</span> {lv.statistic?.toFixed(4) ?? 'N/A'}
                          </div>
                          <div>
                            <span className="text-gray-600">p-value:</span> {lv.p_value?.toFixed(4) ?? 'N/A'}
                          </div>
                          <div>
                            <span className="text-gray-600">Result:</span>{' '}
                            <span className={lv.pass ? 'text-green-600 font-semibold' : 'text-red-600 font-semibold'}>
                              {lv.pass ? 'PASS ✓' : 'FAIL ⚠'}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </CardContent>
              </Card>
            )}
          </div>
        </section>
      )}

      <VivaSenseInstitutional />
    </Layout>
  );
}
