from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (Paragraph, Spacer, Table, TableStyle,
                                 HRFlowable, BaseDocTemplate, Frame, PageTemplate)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY

NVIDIA_GREEN = colors.HexColor("#76B900")
NVIDIA_BLACK = colors.HexColor("#1A1A1A")
NVIDIA_GREY  = colors.HexColor("#555555")
LIGHT_GREY   = colors.HexColor("#F6F6F6")
MID_GREY     = colors.HexColor("#CCCCCC")
GROSS_GRN    = colors.HexColor("#EAF5C8")
TOTAL_GRN    = colors.HexColor("#D4EDA0")

W, H = A4

# Real registered address (MCA verified)
COMPANY_NAME    = "NVIDIA Graphics Private Limited"
COMPANY_ADDR    = "Bagmane Goldstone Building, North Tower, Adjacent to World Technology Centre"
COMPANY_ADDR2   = "Mahadevapura, Marathahalli Outer Ring Road, Bengaluru – 560048, Karnataka"
COMPANY_CIN     = "CIN: U32106KA2004PTC033880"
COMPANY_EMAIL   = "india-hr@nvidia.com"
COMPANY_PARENT  = "A subsidiary of NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, CA 95051, USA"

# Two-pass to get real page count
class PageCounter(BaseDocTemplate):
    def __init__(self, *args, **kwargs):
        BaseDocTemplate.__init__(self, *args, **kwargs)
        self.total_pages = 0
    def handle_pageEnd(self):
        self.total_pages = self.page
        BaseDocTemplate.handle_pageEnd(self)

total_pages_holder = [0]

def make_letterhead(total_pages):
    def letterhead(canv, doc):
        canv.saveState()
        # top green bar
        canv.setFillColor(NVIDIA_GREEN)
        canv.rect(0, H - 20*mm, W, 20*mm, fill=1, stroke=0)
        # NVIDIA wordmark
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 21)
        canv.drawString(19.2*mm, H - 10*mm, "NVIDIA")
        # company name right
        canv.setFont("Helvetica-Bold", 7.5)
        canv.drawRightString(W - 18*mm, H - 9*mm, COMPANY_NAME)
        canv.setFont("Helvetica", 6.8)
        canv.drawRightString(W - 18*mm, H - 13.8*mm,
            f"{COMPANY_ADDR}, {COMPANY_ADDR2}")
        # rule under header
        canv.setStrokeColor(NVIDIA_GREEN)
        canv.setLineWidth(0.5)
        canv.line(18*mm, H - 24*mm, W - 18*mm, H - 24*mm)
        # bottom bar
        canv.setFillColor(NVIDIA_GREEN)
        canv.rect(0, 0, W, 9*mm, fill=1, stroke=0)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica", 6.8)
        canv.drawString(18*mm, 3.2*mm,
            f"{COMPANY_CIN}  |  {COMPANY_PARENT}")
        page_str = f"Page {doc.page} of {total_pages}"
        canv.drawRightString(W - 18*mm, 3.2*mm, page_str)
        canv.restoreState()
    return letterhead

def build_doc(output_path, total_pages=2):
    doc = BaseDocTemplate(
        output_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=30*mm, bottomMargin=16*mm)
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id='body')
    doc.addPageTemplates([PageTemplate(
        id='main', frames=frame,
        onPage=make_letterhead(total_pages))])

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sRef   = S("sRef",  fontName="Helvetica",      fontSize=8,   textColor=NVIDIA_GREY,
               alignment=TA_RIGHT, spaceAfter=2)
    sTitle = S("sTitle",fontName="Helvetica-Bold",  fontSize=13,  textColor=NVIDIA_BLACK,
               alignment=TA_CENTER, spaceAfter=2, spaceBefore=2)
    sBody  = S("sBody", fontName="Helvetica",       fontSize=9,   textColor=NVIDIA_BLACK,
               spaceAfter=3, leading=14, alignment=TA_JUSTIFY)
    sSmall = S("sSmall",fontName="Helvetica",       fontSize=7.5, textColor=NVIDIA_GREY,
               spaceAfter=2, leading=11)
    sLabel = S("sLabel",fontName="Helvetica-Bold",  fontSize=8.5, textColor=NVIDIA_GREY)
    sValue = S("sValue",fontName="Helvetica",        fontSize=8.5, textColor=NVIDIA_BLACK)
    sSec   = S("sSec",  fontName="Helvetica-Bold",  fontSize=8.5, textColor=colors.white, leading=13)
    sSign  = S("sSign", fontName="Helvetica-Bold",  fontSize=8.5, textColor=NVIDIA_BLACK,
               spaceAfter=1, leading=12)
    sSignS = S("sSignS",fontName="Helvetica",        fontSize=8,   textColor=NVIDIA_GREY,
               spaceAfter=1, leading=11)
    sAcc   = S("sAcc",  fontName="Helvetica-Bold",  fontSize=9,   textColor=NVIDIA_GREEN,
               alignment=TA_CENTER, spaceAfter=3)

    def sec(text):
        t = Table([[Paragraph(text, sSec)]], colWidths=[doc.width])
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), NVIDIA_GREEN),
            ('TOPPADDING',    (0,0),(-1,-1), 4),
            ('BOTTOMPADDING', (0,0),(-1,-1), 4),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
        ]))
        return t

    def kv(rows, c1=56*mm):
        data = [[Paragraph(l, sLabel), Paragraph(v, sValue)] for l,v in rows]
        t = Table(data, colWidths=[c1, doc.width - c1])
        t.setStyle(TableStyle([
            ('VALIGN',         (0,0),(-1,-1), 'TOP'),
            ('LEFTPADDING',    (0,0),(-1,-1), 5),
            ('RIGHTPADDING',   (0,0),(-1,-1), 5),
            ('TOPPADDING',     (0,0),(-1,-1), 3),
            ('BOTTOMPADDING',  (0,0),(-1,-1), 3),
            ('ROWBACKGROUNDS', (0,0),(-1,-1), [colors.white, LIGHT_GREY]),
            ('LINEBELOW',      (0,0),(-1,-1), 0.3, MID_GREY),
        ]))
        return t

    story = []

    # ── Ref & date ────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Ref: NV/HR/BLR/2026/LOJ-0312 &nbsp;&nbsp; Date: 01 March 2026", sRef))
    story.append(Spacer(1, 1*mm))

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("LETTER OF JOINING", sTitle))
    story.append(HRFlowable(width="100%", thickness=1.2,
                             color=NVIDIA_GREEN, spaceAfter=5))

    # ── To block ──────────────────────────────────────────────────────────────
    story.append(Paragraph("<b>To,</b>", sBody))
    story.append(Paragraph("<b>Mr. Yash Gupta</b>", sBody))
    story.append(Paragraph("Bareilly, Uttar Pradesh – 243001, India", sBody))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        "<b>Sub: Confirmation of Joining – Cloud Infrastructure Engineer | "
        "NVIDIA Graphics Private Limited, Bengaluru</b>", sBody))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph("Dear Mr. Yash Gupta,", sBody))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        "Further to our selection process and offer communicated to you, we are pleased to confirm "
        "your appointment as <b>Cloud Infrastructure Engineer</b> with "
        "<b>NVIDIA Graphics Private Limited</b>. This letter serves as your formal Letter of Joining "
        "and outlines the key terms of your employment. You are requested to report for duty on "
        "<b>01 April 2026</b> at the Bengaluru office by 9:30 AM.",
        sBody))
    story.append(Spacer(1, 4*mm))

    # ── 1. APPOINTMENT ────────────────────────────────────────────────────────
    story.append(sec("1.   APPOINTMENT DETAILS"))
    story.append(kv([
        ("Designation :",      "Cloud Infrastructure Engineer"),
        ("Department :",       "Cloud Infrastructure Engineering"),
        ("Employee ID :",      "NV-IND-2026-4761"),
        ("Date of Joining :",  "01 April 2026"),
        ("Work Location :",    "Bagmane Goldstone Building, North Tower, Mahadevapura, Bengaluru – 560048"),
        ("Work Mode :",        "Onsite – Monday to Friday"),
        ("Employment Type :",  "Full-Time, Permanent"),
        ("Probation Period :", "6 (Six) months from Date of Joining"),
    ]))
    story.append(Spacer(1, 4*mm))

    # ── 2. COMPENSATION ───────────────────────────────────────────────────────
    story.append(sec("2.   COMPENSATION  (Annual CTC: Rs. 32,00,000)"))

    th_style = S("th", fontName="Helvetica-Bold", fontSize=8, textColor=NVIDIA_BLACK)
    td_style = S("td", fontName="Helvetica",       fontSize=8, textColor=NVIDIA_BLACK)

    ch = [[Paragraph(h, th_style) for h in
           ["Component", "Monthly (Rs.)", "Annual (Rs.)"]]]
    cr = [
        ["Basic Salary",               "1,06,667",  "12,80,000"],
        ["House Rent Allowance (HRA)",   "53,333",   "6,40,000"],
        ["Transport Allowance",           "3,200",     "38,400"],
        ["Special Allowance",            "56,800",   "6,81,600"],
        ["Gross Salary",              "2,20,000",  "26,40,000"],
        ["Employer PF Contribution",     "12,800",   "1,53,600"],
        ["Gratuity Provision",            "5,128",     "61,536"],
        ["Medical & Insurance",             "--",      "44,864"],
        ["Total CTC",                       "--",  "32,00,000"],
    ]
    cdata = ch + [[Paragraph(c, td_style) for c in r] for r in cr]
    cw = [doc.width*0.46, doc.width*0.27, doc.width*0.27]
    ct = Table(cdata, colWidths=cw)
    ct.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor("#F0F7E6")),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, LIGHT_GREY]),
        ('BACKGROUND',    (0,5), (-1,5),  GROSS_GRN),
        ('BACKGROUND',    (0,9), (-1,9),  TOTAL_GRN),
        ('FONTNAME',      (0,5), (-1,5),  'Helvetica-Bold'),
        ('FONTNAME',      (0,9), (-1,9),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('ALIGN',         (1,0), (2,-1),  'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('GRID',          (0,0), (-1,-1), 0.4, MID_GREY),
        ('TOPPADDING',    (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('RIGHTPADDING',  (0,0), (-1,-1), 5),
    ]))
    story.append(ct)
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        "Statutory deductions (Employee PF @ 12% of Basic, Professional Tax, TDS) will be applied "
        "as per applicable law. Monthly salary slip will be provided via the employee portal.", sSmall))
    story.append(Spacer(1, 4*mm))

    # ── 3. KEY TERMS ──────────────────────────────────────────────────────────
    story.append(sec("3.   TERMS OF EMPLOYMENT"))
    story.append(kv([
        ("Working Hours :",     "9 hours per day | 10:00 AM – 07:00 PM IST | Monday to Friday"),
        ("Notice Period :",     "30 days during probation; 60 days post-confirmation, by either party"),
        ("Leave Entitlement :", "Earned Leave: 18 days | Sick Leave: 12 days | Casual Leave: 6 days "
                                "| Public holidays as per Karnataka Government calendar"),
        ("Confidentiality :",   "You are required to execute a Non-Disclosure and IP Assignment Agreement "
                                "on your date of joining. All work product created during employment "
                                "remains the property of the Company."),
        ("Code of Conduct :",   "You will be required to comply with NVIDIA's Global Code of Conduct, "
                                "Information Security Policy, and all applicable Indian labour laws "
                                "including the POSH Act, 2013."),
        ("BGV :",               "This appointment is subject to successful completion of a background "
                                "verification check covering identity, education, and criminal records."),
    ]))
    story.append(Spacer(1, 4*mm))

    # ── 4. DOCUMENTS ──────────────────────────────────────────────────────────
    story.append(sec("4.   DOCUMENTS TO BE SUBMITTED ON JOINING"))
    docs_text = (
        "Proof of Identity (Aadhaar / Passport)  |  PAN Card  |  "
        "Class 10 and Class 12 Marksheets and Certificates  |  "
        "Graduation Degree / Provisional Certificate and all Semester Marksheets  |  "
        "Experience / Internship Letters (if any)  |  "
        "2 Passport-size Photographs  |  Cancelled Cheque  |  "
        "Medical Fitness Certificate from a Registered Physician"
    )
    story.append(Paragraph(docs_text,
        S("dl", fontName="Helvetica", fontSize=8.5, textColor=NVIDIA_BLACK,
          leading=14, spaceAfter=5, leftIndent=0)))
    story.append(Spacer(1, 5*mm))

    # ── Closing ───────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "We look forward to your joining. Should you have any questions, please write to "
        f"<b>{COMPANY_EMAIL}</b>.",
        sBody))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("Yours sincerely,", sBody))
    story.append(Spacer(1, 10*mm))

    # ── Signatures ────────────────────────────────────────────────────────────
    sig = [
        [Paragraph("___________________________", sSign),
         Paragraph("___________________________", sSign)],
        [Paragraph("Sudha Hooda", sSign),
         Paragraph("Vinit Kumar Agarwal", sSign)],
        [Paragraph("Director – Human Resources", sSignS),
         Paragraph("Director", sSignS)],
        [Paragraph("NVIDIA Graphics Private Limited", sSignS),
         Paragraph("NVIDIA Graphics Private Limited", sSignS)],
    ]
    st = Table(sig, colWidths=[doc.width*0.5, doc.width*0.5])
    st.setStyle(TableStyle([
        ('VALIGN',         (0,0),(-1,-1),'TOP'),
        ('TOPPADDING',     (0,0),(-1,-1),2),
        ('BOTTOMPADDING',  (0,0),(-1,-1),2),
        ('LEFTPADDING',    (0,0),(-1,-1),0),
    ]))
    story.append(st)
    story.append(Spacer(1, 7*mm))

    # ── Acceptance ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                             color=NVIDIA_GREEN, spaceAfter=4))
    story.append(Paragraph(
        "ACKNOWLEDGEMENT OF ACCEPTANCE  (To be signed and returned by 31 March 2026)", sAcc))
    story.append(Paragraph(
        "I, <b>Mr. Yash Gupta</b>, hereby acknowledge receipt of this Letter of Joining and "
        "unconditionally accept the terms and conditions stated herein. I confirm that all "
        "information furnished during the recruitment process is accurate, and I undertake to "
        "report for duty on <b>01 April 2026</b>.",
        sBody))
    story.append(Spacer(1, 8*mm))

    acc = [
        [Paragraph("Signature of Candidate: _______________________", sBody),
         Paragraph("Date: ________________", sBody)],
        [Paragraph("Full Name (Print): <b>Yash Gupta</b>", sBody),
         Paragraph("Place: _______________", sBody)],
    ]
    at = Table(acc, colWidths=[doc.width*0.62, doc.width*0.38])
    at.setStyle(TableStyle([
        ('TOPPADDING',   (0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
        ('LEFTPADDING',  (0,0),(-1,-1),0),
    ]))
    story.append(at)
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"Please email the signed copy to {COMPANY_EMAIL} | "
        "Subject: LOJ Acceptance – Yash Gupta – NV-IND-2026-4761",
        sSmall))

    doc.build(story)
    return doc

import os, tempfile

# First pass: get real page count
tmp = tempfile.mktemp(suffix=".pdf")
d = build_doc(tmp)
real_pages = d.page
os.remove(tmp)

# Second pass: render with correct "Page X of N"
OUTPUT = "NVIDIA_Letter_of_Joining_Yash_Gupta.pdf"
build_doc(OUTPUT, total_pages=real_pages)
print(f"Done: {real_pages} pages -> {OUTPUT}")
