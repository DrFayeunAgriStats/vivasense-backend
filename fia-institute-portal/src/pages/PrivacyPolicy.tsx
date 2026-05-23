import React from "react";

const PrivacyPolicy = () => {
  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "2rem 1rem" }}>
      <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 8 }}>
        Privacy Policy
      </h1>
      <p style={{ fontSize: 13, color: "#666", marginBottom: 32 }}>
        VivaSense — Statistical Analysis Platform<br />
        Field-to-Insight Academy (FIA)<br />
        Last Updated: May 2026
      </p>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>1. Introduction</h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          This Privacy Policy describes how VivaSense processes personal data in accordance
          with the Nigeria Data Protection Act 2023 (NDPA 2023). VivaSense is developed
          and operated by Field-to-Insight Academy (FIA), an independent research training
          and capacity development organisation based in Akure, Nigeria. We are committed
          to ensuring that all processing of personal data complies with applicable Nigerian
          data protection laws and internationally recognised security standards.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>2. Scope</h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          This Policy applies to users who upload research datasets for statistical analysis,
          visitors accessing the VivaSense web application, and individuals who communicate
          with us regarding support or enquiries.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>3. Data Controller</h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          Field-to-Insight Academy (FIA) acts as the Data Controller for personal data
          processed through VivaSense. Where third-party infrastructure providers are used,
          they act as Data Processors under contractual obligations ensuring confidentiality
          and security.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>4. Data We Process</h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          <strong>Research data:</strong> When you upload a CSV file, your data is processed
          in volatile server memory for statistical computation only. Research datasets are
          not intentionally stored beyond active processing. Temporary data may exist briefly
          in secure processing memory and is automatically deleted after analysis completion.
          We do not retain raw research datasets.
        </p>
        <p style={{ fontSize: 14, lineHeight: 1.7, marginTop: 8 }}>
          <strong>Technical usage data:</strong> We may collect non-identifiable metadata
          including analysis type, session duration, and error logs solely to improve platform
          performance. We do not collect national identification numbers, biometric data,
          or sensitive personal data.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          5. Legal Basis for Processing
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          Processing is carried out on the following lawful bases under NDPA 2023:
          (1) User consent — when you voluntarily upload data for analysis;
          (2) Performance of a service — processing is necessary to deliver requested
          statistical outputs; (3) Legitimate interests — improving platform reliability
          and security. You may withdraw consent at any time by discontinuing use.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          6. Purpose Limitation
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          We process uploaded data solely for statistical computation, generation of
          analysis outputs, and platform diagnostics. We do not sell data, share research
          data with third parties, use data for advertising, or use uploaded datasets to
          train artificial intelligence models.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          7. AI Interpretation
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          VivaSense uses the Anthropic Claude API to generate natural-language
          interpretations of statistical results. Only computed statistical summaries
          such as means, F-values, and variance components are transmitted — raw datasets
          are never sent to any AI provider. These summaries may be processed by AI
          service providers outside Nigeria under strict data minimisation and
          confidentiality obligations.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          8. Data Retention
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          Uploaded research datasets are retained only for the duration of active
          processing and are automatically deleted after analysis completion. Limited
          system logs may be retained briefly for operational security and troubleshooting
          consistent with industry standards.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          9. Data Security
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          All data transmitted to VivaSense is encrypted using HTTPS/TLS. Our
          infrastructure providers maintain industry-standard security compliance
          frameworks. Temporary files are stored in secure runtime directories and
          deleted after each analysis request.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          10. International Data Transfers
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          Where third-party service providers are located outside Nigeria, appropriate
          safeguards are implemented to ensure compliance with NDPA 2023 cross-border
          transfer requirements, including contractual data protection obligations.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          11. Your Rights Under NDPA 2023
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          Under NDPA 2023 you have the right to access, rectify, or erase your personal
          data, restrict processing, withdraw consent, and lodge complaints with the
          Nigeria Data Protection Commission (NDPC). Because VivaSense does not retain
          uploaded research datasets beyond the session, most user-uploaded data is
          automatically deleted and not stored for subsequent access.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          12. Children
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          VivaSense is not intended for use by minors. We do not knowingly collect
          personal data from children.
        </p>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          13. Changes to This Policy
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          We may update this Policy to reflect legal developments or platform changes.
          Material updates will be published within the platform. Continued use
          constitutes acceptance.
        </p>
      </section>

      <section style={{ marginBottom: 48 }}>
        <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
          14. Contact
        </h2>
        <p style={{ fontSize: 14, lineHeight: 1.7 }}>
          Field-to-Insight Academy (FIA)<br />
          Akure, Nigeria<br />
          Email: info@fieldtoinsightacademy.com.ng<br />
          WhatsApp: +234 902 215 8026<br />
          Website: fieldtoinsightacademy.com.ng
        </p>
      </section>
    </div>
  );
};

export default PrivacyPolicy;
