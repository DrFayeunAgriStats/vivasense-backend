import { forwardRef } from "react";
import { QRCodeSVG } from "qrcode.react";
import type { Certificate } from "@/types/certificate";

interface CertificatePreviewProps {
  certificate: Certificate;
  scale?: number;
}

const FIALogo = ({ size = 80, opacity = 1 }: { size?: number; opacity?: number }) => (
  <svg viewBox="0 0 100 100" width={size} height={size} style={{ opacity }}>
    <defs>
      <linearGradient id="logoBg" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#1E4D2B" />
        <stop offset="100%" stopColor="#1E4D2B" />
      </linearGradient>
    </defs>
    <circle cx="50" cy="50" r="48" fill="url(#logoBg)" stroke="#1E4D2B" strokeWidth="2" />
    <line x1="25" y1="75" x2="25" y2="30" stroke="white" strokeWidth="4" strokeLinecap="round" />
    <path d="M25,55 Q10,50 15,38 Q20,45 25,55" fill="white" opacity="0.9" />
    <path d="M25,42 Q10,35 20,22 Q23,32 25,42" fill="white" opacity="0.9" />
    <path d="M25,50 Q35,42 30,30 Q28,40 25,50" fill="white" opacity="0.7" />
    <circle cx="45" cy="62" r="6" fill="white" opacity="0.9" />
    <circle cx="60" cy="48" r="6" fill="white" opacity="0.9" />
    <circle cx="75" cy="32" r="6" fill="white" opacity="0.9" />
    <polyline points="45,62 60,48 75,32" stroke="#C9A84C" strokeWidth="3" fill="none" strokeLinecap="round" />
  </svg>
);

const DigitalSeal = () => (
  <svg width="80" height="80" viewBox="0 0 140 140" style={{ opacity: 0.92 }}>
    <defs>
      <radialGradient id="sealBg" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stopColor="#f5f3ed" />
        <stop offset="100%" stopColor="#e8e4d8" />
      </radialGradient>
      <linearGradient id="sealGold" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#C9A84C" />
        <stop offset="50%" stopColor="#E8D48B" />
        <stop offset="100%" stopColor="#C9A84C" />
      </linearGradient>
      <linearGradient id="sealGreen" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#1E4D2B" />
        <stop offset="100%" stopColor="#2A6B3C" />
      </linearGradient>
    </defs>
    {/* Outer scalloped/notched ring */}
    {Array.from({ length: 48 }).map((_, i) => {
      const angle = (i * 360) / 48;
      const rad = (angle * Math.PI) / 180;
      const cx = 70 + 67 * Math.cos(rad);
      const cy = 70 + 67 * Math.sin(rad);
      return <circle key={i} cx={cx} cy={cy} r="2.2" fill="#1E4D2B" opacity="0.7" />;
    })}
    {/* Concentric rings */}
    <circle cx="70" cy="70" r="64" fill="none" stroke="#1E4D2B" strokeWidth="2.8" />
    <circle cx="70" cy="70" r="61" fill="url(#sealBg)" stroke="url(#sealGold)" strokeWidth="1.8" />
    <circle cx="70" cy="70" r="57" fill="none" stroke="#1E4D2B" strokeWidth="0.6" />
    <circle cx="70" cy="70" r="55" fill="none" stroke="#1E4D2B" strokeWidth="0.3" strokeDasharray="1.5 1.5" />
    {/* Wheat/leaf wreath — left side */}
    <g opacity="0.75">
      {/* Left branch */}
      <path d="M38,95 Q30,75 38,55" fill="none" stroke="#1E4D2B" strokeWidth="1" />
      <path d="M38,90 Q30,88 28,82" fill="#1E4D2B" opacity="0.5" />
      <path d="M38,84 Q29,82 27,76" fill="#1E4D2B" opacity="0.5" />
      <path d="M38,78 Q30,76 28,70" fill="#1E4D2B" opacity="0.5" />
      <path d="M38,72 Q31,70 30,64" fill="#1E4D2B" opacity="0.5" />
      <path d="M38,66 Q33,64 33,58" fill="#1E4D2B" opacity="0.4" />
      {/* Right branch */}
      <path d="M102,95 Q110,75 102,55" fill="none" stroke="#1E4D2B" strokeWidth="1" />
      <path d="M102,90 Q110,88 112,82" fill="#1E4D2B" opacity="0.5" />
      <path d="M102,84 Q111,82 113,76" fill="#1E4D2B" opacity="0.5" />
      <path d="M102,78 Q110,76 112,70" fill="#1E4D2B" opacity="0.5" />
      <path d="M102,72 Q109,70 110,64" fill="#1E4D2B" opacity="0.5" />
      <path d="M102,66 Q107,64 107,58" fill="#1E4D2B" opacity="0.4" />
    </g>
    {/* Inner decorative ring with fine detail */}
    <circle cx="70" cy="70" r="38" fill="none" stroke="url(#sealGold)" strokeWidth="1.2" />
    <circle cx="70" cy="70" r="36" fill="none" stroke="#1E4D2B" strokeWidth="0.4" strokeDasharray="3 2" />
    {/* Text arcs */}
    <path id="sealTopArc2" d="M 18 70 A 52 52 0 0 1 122 70" fill="none" />
    <path id="sealBottomArc2" d="M 122 70 A 52 52 0 0 1 18 70" fill="none" />
    <text fontSize="5.8" fill="#1E4D2B" fontFamily="'Montserrat', sans-serif" fontWeight="800" letterSpacing="1.6">
      <textPath href="#sealTopArc2" startOffset="50%" textAnchor="middle">FIELD-TO-INSIGHT ACADEMY</textPath>
    </text>
    <text fontSize="5" fill="#1E4D2B" fontFamily="'Montserrat', sans-serif" fontWeight="700" letterSpacing="1">
      <textPath href="#sealBottomArc2" startOffset="50%" textAnchor="middle">CERTIFIED ✦ VERIFIED ✦ OFFICIAL</textPath>
    </text>
    {/* Stars between text bands */}
    <circle cx="18" cy="70" r="2" fill="url(#sealGold)" />
    <circle cx="122" cy="70" r="2" fill="url(#sealGold)" />
    {/* Inner embossed center */}
    <circle cx="70" cy="68" r="22" fill="#1E4D2B" opacity="0.06" />
    <circle cx="70" cy="68" r="20" fill="none" stroke="#1E4D2B" strokeWidth="0.3" />
    <text x="70" y="64" textAnchor="middle" fontSize="14" fontWeight="800" fill="#1E4D2B" fontFamily="'Playfair Display', serif" letterSpacing="1">FIA</text>
    <text x="70" y="76" textAnchor="middle" fontSize="5.5" fontWeight="700" fill="url(#sealGold)" fontFamily="'Montserrat', sans-serif" letterSpacing="1">ADAP™</text>
    {/* Small decorative diamond below */}
    <polygon points="70,82 72,84 70,86 68,84" fill="#C9A84C" opacity="0.7" />
  </svg>
);

export const CertificatePreview = forwardRef<HTMLDivElement, CertificatePreviewProps>(
  ({ certificate, scale = 1 }, ref) => {
    const verifyUrl = `https://field-to-insight-forge.lovable.app/verify?id=${certificate.certificate_id}`;
    const isCompetence = certificate.certificate_type === "Competence";

    const formatDate = (dateStr: string | null) => {
      if (!dateStr) return "";
      return new Date(dateStr).toLocaleDateString("en-GB", {
        day: "numeric",
        month: "long",
        year: "numeric",
      });
    };

    return (
      <div
        ref={ref}
        style={{
          width: 842,
          height: 595,
          transform: `scale(${scale})`,
          transformOrigin: "top left",
          fontFamily: "'Montserrat', sans-serif",
          position: "relative",
          overflow: "hidden",
          background: "#F7F6F2",
        }}
      >
        {/* Double-rule border: outer dark green */}
        <div style={{ position: "absolute", inset: 0, border: "3px solid #1E4D2B", pointerEvents: "none" }} />
        {/* Gold accent line between rules */}
        <div style={{ position: "absolute", inset: 5, border: "0.8px solid #C9A84C", pointerEvents: "none", opacity: 0.55 }} />
        {/* Inner dark green rule */}
        <div style={{ position: "absolute", inset: 9, border: "1.5px solid #1E4D2B", pointerEvents: "none" }} />
        {/* Gold corner accents */}
        <div style={{ position: "absolute", top: 5, left: 5, width: 30, height: 30, borderTop: "2.5px solid #C9A84C", borderLeft: "2.5px solid #C9A84C", pointerEvents: "none" }} />
        <div style={{ position: "absolute", top: 5, right: 5, width: 30, height: 30, borderTop: "2.5px solid #C9A84C", borderRight: "2.5px solid #C9A84C", pointerEvents: "none" }} />
        <div style={{ position: "absolute", bottom: 5, left: 5, width: 30, height: 30, borderBottom: "2.5px solid #C9A84C", borderLeft: "2.5px solid #C9A84C", pointerEvents: "none" }} />
        <div style={{ position: "absolute", bottom: 5, right: 5, width: 30, height: 30, borderBottom: "2.5px solid #C9A84C", borderRight: "2.5px solid #C9A84C", pointerEvents: "none" }} />

        {/* Content */}
        <div
          style={{
            position: "relative",
            zIndex: 1,
            padding: "20px 48px 20px",
            display: "flex",
            flexDirection: "column",
            height: "100%",
          }}
        >
          {/* Spacer to push content down */}
          <div style={{ flex: 1 }} />
          {/* Header */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 14, marginBottom: 8 }}>
            <FIALogo size={44} />
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 18, fontWeight: 800, color: "#1E4D2B", letterSpacing: 2.5, fontFamily: "'Playfair Display', serif" }}>
                Field-to-Insight Academy
              </div>
              <div style={{ fontSize: 7, fontWeight: 600, color: "#555555", letterSpacing: 1.5, marginTop: 2, textTransform: "uppercase" }}>
                Institute for Agricultural Research, Data Analytics & AI Innovation
              </div>
              <div style={{ width: "100%", height: 1, background: "#C9A84C", margin: "6px auto 0", opacity: 0.5 }} />
            </div>
          </div>

          {/* CERTIFICATE title */}
          <div style={{ textAlign: "center", marginBottom: 4 }}>
            <div
              style={{
                fontSize: 38,
                fontWeight: 800,
                color: "#1E4D2B",
                letterSpacing: 6,
                fontFamily: "'Playfair Display', serif",
                textTransform: "uppercase",
                lineHeight: 1,
              }}
            >
              CERTIFICATE
            </div>
            {/* Gold line removed */}
            <div
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: "#1E4D2B",
                textTransform: "uppercase",
                letterSpacing: 5,
                marginTop: 1,
                fontFamily: "'Playfair Display', serif",
                fontVariant: "small-caps",
              }}
            >
              of {certificate.certificate_type}
            </div>
          </div>

          {/* Certify phrase */}
          <div style={{ textAlign: "center", fontSize: 10, color: "#555555", marginBottom: 4, marginTop: 10 }}>
            {certificate.certify_phrase || "This certifies that"}
          </div>

          {/* Recipient Name */}
          <div style={{ textAlign: "center", marginBottom: 8 }}>
            <div style={{ display: "inline-block" }}>
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 700,
                  color: "#1E4D2B",
                  fontFamily: "'Playfair Display', serif",
                  lineHeight: 1.2,
                  wordBreak: "break-word",
                }}
              >
                {certificate.recipient_name}
              </div>
              <div style={{ width: "100%", height: 1.5, background: "#C9A84C", marginTop: 18 }} />
            </div>
          </div>

          {/* Completion phrase + Program */}
          <div style={{ textAlign: "center", marginBottom: 8 }}>
            <div style={{ fontSize: 9.5, color: "#555555", marginBottom: 2 }}>
              {certificate.completion_phrase || "Has satisfactorily fulfilled the requirements of"}
            </div>
            <div
              style={{
                fontSize: 16,
                fontWeight: 700,
                color: "#1E4D2B",
                letterSpacing: 1,
                fontFamily: "'Playfair Display', serif",
              }}
            >
              {certificate.program_name}
            </div>
          </div>

          {/* Competencies with watermark behind */}
          {isCompetence && certificate.competencies && certificate.competencies.length > 0 && (
            <div style={{ position: "relative", marginBottom: 6 }}>
              {/* Watermark behind competency box */}
              <div
                style={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -42%)",
                  opacity: 0.07,
                  pointerEvents: "none",
                  zIndex: 0,
                }}
              >
                <FIALogo size={180} />
              </div>
              <div style={{ position: "relative", zIndex: 1 }}>
                <div
                  style={{
                    fontSize: 8,
                    fontWeight: 700,
                    color: "#1E4D2B",
                    textAlign: "center",
                    textTransform: "uppercase",
                    letterSpacing: 2.5,
                    marginBottom: 3,
                    fontVariant: "small-caps",
                  }}
                >
                  Program Competencies
                </div>
                <div
                  style={{
                    background: "rgba(240, 245, 240, 0.45)",
                    borderTop: "none",
                    borderRight: "none",
                    borderBottom: "none",
                    borderLeft: "1.2px solid #1E4D2B",
                    borderRadius: 0,
                    padding: "8px 20px",
                    maxWidth: 500,
                    margin: "0 auto",
                  }}
                >
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: certificate.competencies.length > 2 ? "1fr 1fr" : "1fr",
                      gap: "6px 28px",
                    }}
                  >
                    {certificate.competencies.map((comp, i) => (
                      <div
                        key={i}
                        style={{
                          textAlign: "center",
                          fontSize: 10,
                          fontWeight: 500,
                          color: "#333",
                          fontFamily: "'Montserrat', sans-serif",
                          padding: "3px 0",
                          lineHeight: 1.3,
                        }}
                      >
                        {comp}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Appreciation text */}
          {!isCompetence && certificate.appreciation_text && (
            <div style={{ textAlign: "center", marginBottom: 4, maxWidth: 500, margin: "0 auto 4px" }}>
              {certificate.appreciation_text.split(",").map((line, i) => (
                <div key={i} style={{ fontSize: 10, fontWeight: 500, color: "#555555", lineHeight: 1.5, marginBottom: 1 }}>
                  {line.trim()}
                </div>
              ))}
            </div>
          )}


          {/* Issuance statement */}
          <div style={{ textAlign: "center", fontSize: 8.5, fontStyle: "italic", color: "#555555", marginBottom: 6, marginTop: 4 }}>
            Issued upon successful completion of structured training and practical assessment requirements
          </div>

          {/* Signature Block */}
          <div style={{ textAlign: "center", marginTop: 0, marginBottom: 0 }}>
            {certificate.signature_url ? (
              <img
                src={certificate.signature_url}
                alt="Signature"
                style={{ height: 28, width: "auto", opacity: 0.85, display: "block", margin: "0 auto 2px" }}
                crossOrigin="anonymous"
              />
            ) : (
              <div style={{ height: 28, marginBottom: 2 }} />
            )}
            <div style={{ width: "55%", height: 0.8, background: "#333", margin: "0 auto 4px", maxWidth: 220 }} />
            <div style={{ fontSize: 12, fontWeight: 700, color: "#1A1A1A", fontFamily: "'Playfair Display', serif" }}>
              Dr. Fayeun Lawrence Stephen
            </div>
            <div style={{ fontSize: 7.5, fontWeight: 600, color: "#555555", marginTop: 2 }}>
              Director, Field-to-Insight Academy
            </div>
          </div>

          {/* Spacer to push footer down */}
          <div style={{ flex: 1 }} />

          {/* Bottom: 3-column */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%" }}>
            {/* Left: details */}
            <div
              style={{
                minWidth: 180,
                maxWidth: 180,
                fontSize: 8,
                color: "#555555",
                lineHeight: 2.2,
              }}
            >
              <div style={{ whiteSpace: "nowrap" }}>
                <span style={{ fontWeight: 700, color: "#444" }}>Certificate No: </span>
                <span style={{ fontFamily: "monospace", fontSize: 8 }}>{certificate.certificate_id}</span>
              </div>
              {certificate.training_start_date && certificate.training_end_date && (
                <div style={{ whiteSpace: "nowrap" }}>
                  <span style={{ fontWeight: 700, color: "#444" }}>Training Period: </span>
                  <span>{formatDate(certificate.training_start_date)} – {formatDate(certificate.training_end_date)}</span>
                </div>
              )}
              <div>
                <span style={{ fontWeight: 700, color: "#444" }}>Date Issued: </span>
                <span>{formatDate(certificate.issue_date)}</span>
              </div>
            </div>

            {/* Center: Seal — mathematically centered */}
            <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center" }}>
              <div style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
                <DigitalSeal />
              </div>
            </div>

            {/* Right: QR */}
            <div style={{ minWidth: 180, maxWidth: 180, display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
              <div
                style={{
                  border: "1.5px solid #C9A84C",
                  borderRadius: 8,
                  padding: 5,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <QRCodeSVG value={verifyUrl} size={48} level="H" />
              </div>
              <div
                style={{
                  fontSize: 6.5,
                  fontWeight: 700,
                  color: "#1E4D2B",
                  letterSpacing: 2,
                  textTransform: "uppercase",
                  marginTop: 1,
                }}
              >
                Scan to Verify
              </div>
              <div style={{ fontSize: 6, color: "#777", lineHeight: 1.3, textAlign: "center", maxWidth: 110 }}>
                www.fieldtoinsightacademy.com/verify
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
);

CertificatePreview.displayName = "CertificatePreview";
