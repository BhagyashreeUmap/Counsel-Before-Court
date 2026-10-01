"""Citizen-facing Streamlit UI. Run: streamlit run app.py."""
import logging

import streamlit as st
from ui_styles import CSS, LOGO, MOTIF, header

from backend.pipeline import CasePipeline
from backend.private_history import PrivateHistory
from backend import memory
from ui_helpers import create_case, workflow_view, gap_responses, progress, current_activity, listener_recovery

# Static labels only. Generated content is never retranslated by the UI.
LABELS = {
    "tagline": (
    "Understand your case. Prepare before you meet a lawyer.",
    "तुमची परिस्थिती समजून घ्या. वकिलाला भेटण्यापूर्वी तयारी करा.",
),

"guide": (
    "Your guide before you meet a lawyer",
    "वकिलाला भेटण्यापूर्वी तुमचा मार्गदर्शक",
),

"intro": (
    "Turn your land-dispute story into organized facts, evidence gaps, similar-case insights, and a lawyer-ready preparation plan.",
    "तुमच्या जमीन वादाची माहिती व्यवस्थित तथ्ये, पुराव्यातील उणिवा, समान प्रकरणांतील माहिती आणि वकिलासाठी तयार केलेल्या तयारीच्या आराखड्यात बदला.",
),
    "features": (
    "Understand what information matters|Know what documents to gather or verify|Learn from similar cases|Prepare specific questions for a lawyer",
    "कोणती माहिती महत्त्वाची आहे ते समजा|कोणती कागदपत्रे गोळा किंवा पडताळायची ते जाणून घ्या|समान प्रकरणांमधून शिका|वकिलाला विचारण्यासाठी विशिष्ट प्रश्न तयार करा",
),
    "access": ("Profile Access", "प्रोफाइल प्रवेश"),
    "name": ("Display name", "दर्शवायचे नाव"), "contact": ("Email OR mobile number", "ईमेल किंवा मोबाइल क्रमांक"),
    "privacy": ("Before you begin", "सुरुवात करण्यापूर्वी"),
    "privacy_text": ("Your contact recovers your private history. Private cases are not automatically added to shared learning. After a real-world resolution, you may separately choose an anonymized contribution: the prototype removes personal information, shows the exact preview, and saves only after your explicit confirmation. Anonymization is not guaranteed complete.", "तुमच्या संपर्कामुळे खाजगी इतिहास पुन्हा उघडता येतो. खाजगी प्रकरणे आपोआप सामायिक शिक्षणात जोडली जात नाहीत. प्रत्यक्ष प्रकरण निकाली निघाल्यावर तुम्ही स्वतंत्रपणे अनामिक योगदान निवडू शकता: नमुना वैयक्तिक माहिती काढतो, अचूक पूर्वावलोकन दाखवतो आणि तुमच्या स्पष्ट पुष्टीनंतरच जतन करतो. संपूर्ण अनामिकीकरणाची हमी नाही."),
    "prototype": ("Prototype access does not verify identity. Anyone knowing your contact may recover the profile. Saved data is not encrypted; case facts are sent to the configured AI provider when you request analysis.", "नमुना प्रवेश ओळख पडताळत नाही. तुमचा संपर्क माहीत असलेली व्यक्ती प्रोफाइल उघडू शकते. जतन केलेला डेटा कूटबद्ध नाही; विश्लेषण मागितल्यावर तथ्ये निवडलेल्या AI पुरवठादाराकडे पाठवली जातात."),
    "ack": ("I understand how my case information is handled.", "माझ्या प्रकरणाची माहिती कशी हाताळली जाते हे मला समजले."),
    "continue": ("Continue", "पुढे जा"), "cases": ("My Cases", "माझी प्रकरणे"),
    "welcome": ("Welcome", "स्वागत"), "new": ("Start a New Case", "नवीन प्रकरण सुरू करा"),
    "title": ("Case title", "प्रकरणाचे शीर्षक"), "default_title": ("My case preparation", "माझ्या प्रकरणाची तयारी"),
    "empty": ("No saved cases yet. Start a case to begin preparing your situation.", "अद्याप प्रकरणे जतन केलेली नाहीत. तयारीसाठी नवीन प्रकरण सुरू करा."),
    "open": ("Open / Resume", "उघडा / पुढे सुरू करा"), "delete": ("Delete saved private case", "जतन केलेले खाजगी प्रकरण हटवा"),
    "delete_note": ("This deletes this saved private case from My Cases. It does not delete any previously confirmed anonymized contribution.", "यामुळे माझी प्रकरणे मधील हे खाजगी प्रकरण हटते. पूर्वी पुष्टी केलेले अनामिक योगदान हटत नाही."),
    "confirm_delete": ("Confirm deletion of this private case", "हे खाजगी प्रकरण हटवण्याची पुष्टी करा"),
    "workspace": ("Case Workspace", "प्रकरण कार्यक्षेत्र"), "progress": ("Case Progress", "प्रकरणाची प्रगती"),
    "activity": ("Agent Activity", "एजंटची कार्यवाही"), "message": ("Type your message…", "तुमचा संदेश लिहा…"),
    "send": ("Send", "पाठवा"), "processing": ("Preparing your case…", "तुमच्या प्रकरणाची तयारी सुरू आहे…"),
    "resume": ("Continue preparation", "तयारी पुढे सुरू करा"),
    "error": ("We couldn't continue this step. Your previous saved case remains available.", "हे पाऊल पुढे नेता आले नाही. पूर्वी जतन केलेले प्रकरण उपलब्ध आहे."),
    "save_error": ("This progress is still in this session but could not be saved. Retry saving before leaving.", "ही प्रगती सत्रात आहे पण जतन झाली नाही. बाहेर पडण्यापूर्वी पुन्हा जतन करा."),
    "retry_save": ("Retry saving", "पुन्हा जतन करा"),
    "intake_error": ("We couldn't complete this intake turn. No new facts were accepted. You can return to the previous intake state and submit a new answer.", "हे माहिती संकलनाचे पाऊल पूर्ण झाले नाही. नवीन तथ्ये स्वीकारली नाहीत. मागील स्थितीत परत जाऊन नवीन उत्तर पाठवू शकता."),
    "recover_intake": ("Return to previous intake state", "मागील माहिती संकलन स्थितीत परत जा"),
    "event_history": ("Previous activity events (historical, not current statuses)", "पूर्वीची कार्यवाही (इतिहास, सध्याच्या स्थिती नाहीत)"),
    "summary": ("Case Summary", "प्रकरणाचा सारांश"), "observations": ("Key observations", "महत्त्वाची निरीक्षणे"),
    "why_ask": ("Why we ask", "आम्ही का विचारतो"),
    "you": ("You", "तुम्ही"),
    "story_empty": ("Describe what happened in your own words. Start with the people involved and what you want to understand.", "काय घडले ते तुमच्या शब्दांत सांगा. कोणाचा संबंध आहे आणि तुम्हाला काय समजून घ्यायचे आहे यापासून सुरुवात करा."),
    "complete_badge": ("Preparation complete", "तयारी पूर्ण"), "evidence_badge": ("Evidence needed", "पुराव्याची माहिती हवी"), "progress_badge": ("In progress", "प्रगतीत"),
    "land_dispute": ("Land dispute", "जमिनीचा वाद"),
    "safety": ("Need immediate help?", "तातडीची मदत हवी आहे का?"),
    "safety_text": ("Your story indicates an immediate safety concern. Ordinary case analysis has paused. Consider seeking support from someone you trust.", "तुमच्या कथेत तातडीच्या सुरक्षिततेची चिंता दिसते. सामान्य विश्लेषण थांबले आहे. विश्वासू व्यक्तीची मदत घेण्याचा विचार करा."),
    "support": ("View Support Options", "मदतीचे पर्याय पाहा"),
    "no_support": ("No grounded support listing is available for this case. No emergency numbers are supplied by this prototype.", "या प्रकरणासाठी आधार असलेली मदत नोंद उपलब्ध नाही. हा नमुना आपत्कालीन क्रमांक पुरवत नाही."),
    "evidence": ("We need a few more details", "आणखी थोडी माहिती हवी आहे"),
    "evidence_note": ("These details help us prepare your case more accurately. Explanations are supplementary; your Yes / No / Not sure selection determines availability.", "ही माहिती अधिक अचूक तयारीसाठी मदत करते. स्पष्टीकरण पूरक आहे; होय / नाही / खात्री नाही ही निवड उपलब्धता ठरवते."),
    "available": ("Yes, I have it", "होय, माझ्याकडे आहे"), "unavailable": ("No", "नाही"), "unknown": ("Not sure", "खात्री नाही"),
    "explanation": ("Optional explanation", "ऐच्छिक स्पष्टीकरण"), "submit": ("Submit answers", "उत्तरे पाठवा"),
    "preparation": ("Your Case Preparation", "तुमच्या प्रकरणाची तयारी"),
    "tabs": ("Overview|Documents|Similar Cases|Options|Questions|Legal Help", "आढावा|कागदपत्रे|समान प्रकरणे|पर्याय|प्रश्न|कायदेशीर मदत"),
    "mentioned": ("Mentioned by you", "तुम्ही सांगितलेले"), "check_if_available": ("Check if available", "उपलब्ध आहे का तपासा"),
    "synthetic": ("Synthetic sample; not court precedent", "कृत्रिम नमुना; न्यायालयीन उदाहरण नाही"),
    "user_contributed_anonymized": ("Anonymized citizen-reported experience; unverified", "नागरिकाने सांगितलेला अनामिक अनुभव; पडताळलेला नाही"),
    "options_note": ("Possible next steps to discuss or consider", "चर्चा किंवा विचार करण्याचे संभाव्य पुढील पर्याय"),
    "benefit": ("Possible benefit", "संभाव्य फायदा"), "tradeoff": ("Possible tradeoff", "संभाव्य मर्यादा"),
    "referrals": ("Example listings for demonstration purposes. Nothing is contacted automatically. No eligibility assessment is made.", "प्रात्यक्षिकासाठी काल्पनिक नोंदी. आपोआप कोणाशीही संपर्क होत नाही. पात्रतेचे मूल्यांकन केलेले नाही."),
    "legal_aid": ("Legal aid", "कायदेशीर मदत"), "lawyers": ("Lawyers", "वकील"),
    "no_referrals": ("No referral was returned for this case.", "या प्रकरणासाठी संदर्भ मिळाले नाहीत."),
    "download": ("Download Lawyer-Ready Summary", "वकिलांसाठी तयार सारांश डाउनलोड करा"),
    "completion": ("Case Completion / Optional Anonymous Learning", "प्रकरणाचा निकाल / ऐच्छिक अनामिक शिक्षण"),
    "resolved_question": ("Has your real-world case been resolved since using Counsel Before Court? Preparation being complete does not mean your dispute is resolved.", "Counsel Before Court वापरल्यानंतर तुमचे प्रत्यक्ष प्रकरण निकाली निघाले आहे का? तयारी पूर्ण झाली म्हणजे वाद निकाली निघाला असे नाही."),
    "not_yet": ("Not yet", "अद्याप नाही"), "resolved": ("Yes, it has been resolved", "होय, निकाली निघाले आहे"),
    "private": ("Keep Private", "खाजगी ठेवा"), "share": ("Share Anonymously", "अनामिक योगदान द्या"),
    "private_note": ("Your case stays in private history. No new shared learning is created. Previously confirmed contributions are unaffected.", "तुमचे प्रकरण खाजगी इतिहासात राहते. नवीन सामायिक शिक्षण तयार होत नाही. पूर्वी पुष्टी केलेल्या योगदानावर परिणाम होत नाही."),
    "share_note": ("Review an anonymized version before deciding to contribute. Review carefully for remaining personal information.", "योगदानाचा निर्णय घेण्यापूर्वी अनामिक आवृत्ती पाहा. उरलेली वैयक्तिक माहिती काळजीपूर्वक तपासा."),
    "outcome": ("How was the case resolved?", "प्रकरणाचा निकाल कसा लागला?"),
    "favourable": ("Favourable", "अनुकूल"), "settled": ("Settled", "समेट"), "unfavourable": ("Unfavourable", "प्रतिकूल"),
    "months": ("Time taken (months)", "लागलेला वेळ (महिने)"), "cost": ("Reported cost level", "सांगितलेली खर्च पातळी"),
    "low": ("Low", "कमी"), "medium": ("Medium", "मध्यम"), "high": ("High", "जास्त"),
    "tried": ("Options actually tried — one per line", "प्रत्यक्ष वापरलेले पर्याय — प्रत्येक ओळीत एक"),
    "factors": ("Reported key factors — one per line", "सांगितलेले महत्त्वाचे घटक — प्रत्येक ओळीत एक"),
    "lesson": ("What did you learn?", "तुम्ही काय शिकलात?"),
    "preview": ("Preview Anonymized Version", "अनामिक आवृत्तीचे पूर्वावलोकन"),
    "exact": ("This is exactly what will be saved.", "नेमके हेच जतन केले जाईल."),
    "consent": ("I consent to contribute this anonymized version to shared case learning.", "ही अनामिक आवृत्ती सामायिक शिक्षणात देण्यास माझी संमती आहे."),
    "confirm_share": ("Confirm & Share Anonymously", "पुष्टी करा आणि अनामिक योगदान द्या"),
    "thanks": ("Thank you. Your anonymized experience can help future citizens. Your private history remains saved.", "धन्यवाद. तुमचा अनामिक अनुभव भविष्यात नागरिकांना मदत करू शकतो. खाजगी इतिहास जतन आहे."),
    "unsupported": ("Anonymous learning currently supports completed land-dispute cases only. Your private case remains saved.", "अनामिक शिक्षण सध्या निकाली निघालेल्या जमिनीच्या वादांसाठीच उपलब्ध आहे. खाजगी प्रकरण जतन आहे."),
    "completed": ("Completed", "पूर्ण"), "current": ("Current", "सध्या"), "waiting": ("Waiting", "प्रतीक्षेत"), "not_required": ("Not required", "आवश्यक नाही"),
    "story_listener": ("Your Story", "तुमची कथा"), "research": ("Research", "संशोधन"), "analyst_initial": ("Case Analysis", "प्रकरण विश्लेषण"),
    "gap": ("Evidence Check", "पुरावा तपासणी"), "analyst_final": ("Final Analysis", "अंतिम विश्लेषण"), "guidance": ("Guidance", "मार्गदर्शन"),
}


def t(key):
    return LABELS.get(key, (key, key))[st.session_state.get("language", "English") == "मराठी"]


def safe_action(action):
    try:
        action()
        return True
    except Exception as exc:
        # Log technical details only in terminal
        logging.getLogger(__name__).warning(
            "UI action failed: %s", type(exc).__name__
        )

       
        st.toast(t("error"), icon="⚠️")
        return False


def persist(store):
    s = st.session_state
    store.save_private_case_snapshot(s.profile["id"], s.case_id, s.pipeline.state())
    s.dirty = False


def changed(store):
    s = st.session_state
    s.dirty = True
    safe_action(lambda: persist(store))
    s.view = workflow_view(s.pipeline.state())


def reset_learning():
    for key in list(st.session_state):
        if key.startswith("learn_"):
            del st.session_state[key]
    st.session_state.preview = None
    st.session_state.shared = None


def welcome(store):
    left, right = st.columns([.58, .42], gap="medium")

    with left, st.container(key="welcome-story", gap="small"):
        st.caption("CITIZEN LEGAL PREPARATION")
        st.title(t("tagline"))
        st.write(t("intro"))

        st.markdown(
            (
                "**Tell your story**  →  **Find similar cases**  →  "
                "**Check evidence**  →  **Prepare next steps**"
            )
            if s_language_english()
            else (
                "**तुमची बाजू सांगा**  →  **समान प्रकरणे शोधा**  →  "
                "**पुरावे तपासा**  →  **पुढील तयारी करा**"
            )
        )

        st.divider()

        features = t("features").split("|")
        icons = ("fact_check", "folder_open", "search", "question_answer")

        for offset in (0, 2):
            columns = st.columns(2, gap="medium")
            for column, text, icon in zip(
                columns,
                features[offset:offset + 2],
                icons[offset:offset + 2],
            ):
                with column:
                    st.markdown(f":green[:material/{icon}:] **{text}**")

        st.image(MOTIF, width=430)

        with st.expander(
            "Privacy & prototype limitations"
            if s_language_english()
            else "गोपनीयता आणि नमुन्याच्या मर्यादा",
            icon=":material/privacy_tip:",
        ):
            st.write(t("privacy_text"))
            st.write(t("prototype"))

    with right, st.container(border=True, key="welcome-profile", gap="xsmall"):
        english = s_language_english()
        labels = [
            ("PRIVATE HISTORY", "Your saved cases are not added to shared learning."),
            ("OPTIONAL LEARNING", "Only resolved experiences you choose can be considered for anonymous sharing."),
            ("YOU REVIEW FIRST", "You see the exact anonymized record before confirming."),
        ]

        st.subheader("Privacy at a glance" if english else "गोपनीयता एका नजरेत")

        for icon, (title, body) in zip(
            ("lock", "shield", "visibility"),
            labels,
        ):
            st.markdown(f":green[:material/{icon}:] **{title}**")
            st.caption(body)

        acknowledged = st.checkbox(t("ack"), key="privacy_ack")

        with st.form("profile_access", border=False):
            st.subheader(t("access"))
            name = st.text_input(t("name"))
            contact = st.text_input(t("contact"))
            submitted = st.form_submit_button(
                t("continue"),
                disabled=not acknowledged,
                type="primary",
                width="stretch",
                icon=":material/arrow_forward:",
            )

    if submitted and acknowledged:
        def access():
            st.session_state.profile = store.get_or_create_profile(name, contact)
            st.session_state.view = "cases"

        safe_action(access)
        st.rerun()
        st.write(t("privacy_text"))
        st.write(t("prototype"))
    with right, st.container(border=True, key="welcome-profile", gap="xsmall"):
        english = s_language_english()
        labels = [("PRIVATE HISTORY", "Your saved cases are not added to shared learning."), ("OPTIONAL LEARNING", "Only resolved experiences you choose can be considered for anonymous sharing."), ("YOU REVIEW FIRST", "You see the exact anonymized record before confirming.")] if english else [("खाजगी इतिहास", "जतन केलेली प्रकरणे सामायिक शिक्षणात जोडली जात नाहीत."), ("ऐच्छिक शिक्षण", "तुम्ही निवडलेले निकाली निघालेले अनुभवच अनामिक योगदानासाठी विचारात घेतले जातात."), ("आधी तुमचे पुनरावलोकन", "पुष्टी करण्यापूर्वी अचूक अनामिक नोंद पाहता येते.")]
        st.subheader("Privacy at a glance" if english else "गोपनीयता एका नजरेत")
        for icon, (title, body) in zip(("lock", "shield", "visibility"), labels):
            st.markdown(f":green[:material/{icon}:] **{title}**")
            st.caption(body)
        acknowledged = st.checkbox(t("ack"), key="privacy_ack")
        with st.form("profile_access", border=False):
            st.subheader(t("access"))
            name = st.text_input(t("name"))
            contact = st.text_input(t("contact"))
            submitted = st.form_submit_button(t("continue"), disabled=not acknowledged, type="primary", width="stretch", icon=":material/arrow_forward:")
    if submitted and acknowledged:
        def access():
            st.session_state.profile = store.get_or_create_profile(name, contact)
            st.session_state.view = "cases"
        if safe_action(access):
            st.rerun()


def s_language_english():
    """Static presentation copy follows the existing language selection."""
    return st.session_state.get("language", "English") == "English"


def my_cases(store):
    s = st.session_state
    st.header(t("cases"))
    st.write(f"{t('welcome')}, {s.profile['display_name']}")
    with st.form("new_case", border=False):
        title_column, action_column = st.columns([4, 1], vertical_alignment="bottom")
        title = title_column.text_input(t("title"), value=t("default_title"))
        with action_column:
            new = st.form_submit_button(t("new"), type="primary", disabled=s.get("dirty", False), width="stretch", icon=":material/add:")
    if new:
        def start():
            s.case_id, s.pipeline = create_case(store, s.profile["id"], title)
            s.dirty = False
            reset_learning()
            s.view = "workspace"
        if safe_action(start):
            st.rerun()
    records = store.get_profile_cases(s.profile["id"])
    if not records:
        st.caption(t("empty"))
    for record in records:
        with st.container(border=True):
            badge = t("complete_badge" if record["status"] == "COMPLETE" else "evidence_badge" if record["status"] == "WAITING_FOR_GAP_ANSWERS" else "progress_badge")
            info_column, action_column = st.columns([4, 1], vertical_alignment="center")
            info_column.subheader(record["title"])
            info_column.caption(f"{t(record['legal_area'])} · {record['updated_at'][:10]}")
            info_column.badge(badge, color="green" if record["status"] == "COMPLETE" else "gray")
            if action_column.button(t("open"), key="open_" + record["id"], disabled=s.get("dirty", False), width="stretch", icon=":material/arrow_forward:"):
                def reopen():
                    s.pipeline = store.reopen_private_case(s.profile["id"], record["id"])
                    s.case_id = record["id"]
                    s.dirty = False
                    reset_learning()
                    s.view = workflow_view(s.pipeline.state())
                if safe_action(reopen):
                    st.rerun()
            with st.expander(t("delete"), icon=":material/delete:", type="compact"):
                st.write(t("delete_note"))
                confirm = st.checkbox(t("confirm_delete"), key="delete_ack_" + record["id"])
                if st.button(t("delete"), key="delete_" + record["id"], disabled=not confirm):
                    if safe_action(lambda: store.delete_private_case(s.profile["id"], record["id"])):
                        if s.get("case_id") == record["id"]:
                            s.pop("pipeline", None)
                            s.pop("case_id", None)
                            reset_learning()
                        st.rerun()


def activity_steps(snapshot):
    """Read-only presentation of observable results; never infer agent actions."""
    latest = {event["agent"]: event for event in current_activity(snapshot)}
    completed = {event["agent"] for event in snapshot["activity_events"] if event["status"] == "completed"}
    case = snapshot["case_file"]
    mr = st.session_state.get("language") == "मराठी"
    def label(en, translated):
        return translated if mr else en
    names = [label("Story Listener", "कथा संकलन एजंट"), label("Research Agent", "संशोधन एजंट"),
             label("Case Analyst", "प्रकरण विश्लेषक"), label("Evidence Gap Agent", "पुरावा तपासणी एजंट"),
             label("Final Analyst", "अंतिम विश्लेषक"), label("Guidance Agent", "मार्गदर्शन एजंट")]
    pending = [label("Structures your citizen story", "तुमच्या कथेतून तथ्ये संकलित करतो"),
               label("Searches shared case memory", "सामायिक प्रकरण स्मृतीत शोध घेतो"),
               label("Compares case patterns and evidence", "प्रकरणांचे नमुने आणि पुरावे तपासतो"),
               label("Checks missing evidence", "न सांगितलेले पुरावे तपासतो"),
               label("Re-analyzes after confirmed evidence", "पुष्टी केलेल्या पुराव्यानंतर पुन्हा विश्लेषण करतो"),
               label("Builds lawyer-ready preparation", "वकिलांना भेटण्यासाठी तयारी करतो")]
    for index, (agent, progress_status) in enumerate(progress(snapshot)):
        event = latest.get(agent, {})
        status = event.get("status", "waiting" if progress_status == "current" else progress_status)
        lines = []
        if agent == "story_listener":
            if agent in completed and any(case.get(field) for field in ("parties_and_relationship", "property_or_matter_details", "timeline", "other_side_claim")):
                lines.append(label("Structured citizen story", "नागरिकाच्या कथेतून तथ्ये संकलित केली"))
            if agent in completed and case.get("citizen_goal"):
                lines.append(label("Recorded citizen-stated goal", "नागरिकाने सांगितलेले उद्दिष्ट नोंदवले"))
        elif agent == "research" and snapshot.get("research_result") is not None:
            lines.append(label("Completed shared-memory case search", "सामायिक स्मृतीतील प्रकरणांचा शोध पूर्ण केला"))
            for match in snapshot["research_result"].get("matches", []):
                lines.append(label("Retrieved similar case: ", "समान प्रकरण मिळाले: ") + match["id"])
        elif agent == "analyst_initial" and snapshot.get("initial_analysis") is not None:
            lines.append(label("Compared retrieved case patterns/evidence", "मिळालेल्या प्रकरणांचे नमुने आणि पुरावे तपासले"))
            if (snapshot.get("gap_result") or {}).get("missing_evidence"):
                lines.append(label("Evidence gaps recorded by Gap Agent", "पुरावा तपासणी एजंटने उणिवा नोंदवल्या"))
        elif agent == "gap" and snapshot.get("gap_result") is not None:
            lines.append(label("Checked missing evidence", "न सांगितलेल्या पुराव्यांची तपासणी केली"))
            if snapshot.get("gap_answers"):
                lines.append(label("Recorded structured citizen responses", "नागरिकांची संरचित उत्तरे नोंदवली"))
        elif agent == "analyst_final":
            if agent in completed and snapshot.get("final_analysis") is not None:
                lines.append(label("Re-analyzed updated case evidence", "नवीन पुराव्यांसह पुन्हा विश्लेषण केले"))
            elif progress_status == "not_required":
                lines.append(label("Initial analysis retained; no second pass required", "प्रारंभिक विश्लेषण कायम; दुसरी फेरी आवश्यक नव्हती"))
        elif agent == "guidance" and snapshot.get("guidance_result") is not None:
            lines.append(label("Prepared lawyer-ready summary and options", "वकिलांसाठी सारांश आणि पर्याय तयार केले"))
        waiting = None
        if status == "waiting" and agent == "story_listener" and snapshot["current_stage"] == "LISTENING":
            waiting = label("Waiting for your story response", "तुमच्या उत्तराची प्रतीक्षा")
        if agent == "gap" and snapshot["current_stage"] == "WAITING_FOR_GAP_ANSWERS":
            waiting = label(f"Waiting for {len(snapshot['gap_items'])} citizen responses", f"नागरिकांच्या {len(snapshot['gap_items'])} उत्तरांची प्रतीक्षा")
        yield names[index], status, lines, waiting or (event.get("message") if status == "error" else None), pending[index]


def activity(snapshot, show_history=True):
    st.subheader(t("activity"))
    with st.container(height=390, border=False, gap="small", autoscroll=False):
        for name, status, lines, notice, pending in activity_steps(snapshot):
            icon = "check_circle" if status == "completed" else "radio_button_checked" if notice or status in ("started", "current") else "radio_button_unchecked"
            color = "green" if status == "completed" else "orange" if notice or status in ("started", "current") else "gray"
            with st.container(gap=None):
                st.markdown(f":{color}[:material/{icon}:] **{name}**")
                for line in lines:
                    st.caption(":material/check: " + line)
                if notice:
                    st.caption(notice)
                elif not lines:
                    st.caption(pending)
    if show_history:
        with st.expander(t("event_history"), type="compact"):
            activity_history(snapshot)


def activity_history(snapshot):
    """Render the existing event history without nesting expanders."""
    for index, event in enumerate(snapshot["activity_events"], 1):
        status = "turn finished" if event["status"] == "completed" else event["status"]
        st.caption(f"{index}. {t(event['agent'])} — {status}: {event['message']}")


def context(snapshot):
    st.subheader(t("progress"))
    for index, (agent, status) in enumerate(progress(snapshot), 1):
        icon = "check_circle" if status == "completed" else "radio_button_checked" if status == "current" else "radio_button_unchecked"
        color = "green" if status == "completed" else "orange" if status == "current" else "gray"
        with st.container(gap=None):
            st.markdown(f":{color}[:material/{icon}:] **{index}. {t(agent)}**")
            st.caption(t(status))


def workspace(store, evidence=False):
    s = st.session_state
    snapshot = s.pipeline.state()
    st.header(t("evidence" if evidence else "workspace"))
    left, center, right = st.columns([.17, .55, .28], gap="small")
    with left, st.container(key="progress-rail", gap="xxsmall"):
        context(snapshot)
    with right, st.container(key="agent-activity", gap="small"):
        activity(snapshot)
    with center, st.container(key="case-conversation", gap="xsmall"):
        if snapshot["case_file"].get("danger_flag"):
            st.markdown(f"**{t('safety')}**")
            st.warning(t("safety_text"), icon=":material/warning:")
            with st.expander(t("support")):
                st.write(t("no_support"))
            return
        if snapshot["current_stage"] == "ERROR":
            recovery = listener_recovery(snapshot)
            st.error(t("intake_error" if recovery is not None else "error"))
            if recovery is not None and st.button(t("recover_intake")):
                s.pipeline = CasePipeline(recovery)
                changed(store)
                st.rerun()
            return
        if snapshot["current_stage"] not in ("LISTENING", "WAITING_FOR_GAP_ANSWERS"):
            if st.button(t("resume"), type="primary"):
                with st.spinner(t("processing")):
                    ok = safe_action(s.pipeline.resume)
                if ok:
                    changed(store)
                    st.rerun()
            return
        if evidence:
            st.subheader(t("why_ask"))
            st.caption(t("evidence_note"))
            with st.form("gap_" + s.case_id, border=False):
                statuses, explanations = [], []
                question_columns = st.columns(2)
                for index, item in enumerate(snapshot["gap_items"]):
                    with question_columns[index % 2], st.container(border=True, gap="xsmall"):
                        st.subheader(item["question"], icon=":material/description:")
                        st.caption(item["evidence"])
                        statuses.append(st.segmented_control(item["question"], ["unknown", "available", "unavailable"], format_func=t,
                                                 default=None if s.case_id + item["id"] in s else "unknown", required=True, key=s.case_id + item["id"], label_visibility="collapsed"))
                        explanations.append(st.text_input(t("explanation"), key="explain_" + s.case_id + item["id"]))
                submit = st.form_submit_button(t("submit"), type="primary", icon=":material/arrow_forward:")
            if submit:
                with st.spinner(t("processing")):
                    ok = safe_action(lambda: s.pipeline.gap_answers(gap_responses(snapshot["gap_items"], statuses, explanations)))
                if ok:
                    changed(store)
                    st.rerun()
        else:
            st.subheader(t("story_listener"), icon=":material/chat_bubble_outline:")
            with st.container(height=340 if snapshot["conversation"] else 160, border=True, gap="xsmall"):
                if not snapshot["conversation"]:
                    st.caption(t("story_empty"))
                for message in snapshot["conversation"]:
                    with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/forum:"):
                        st.caption(t("you") if message["role"] == "user" else "Counsel Before Court")
                        st.write(message["content"])
            message = st.chat_input(t("message"), key="story_" + s.case_id)
            if message:
                with st.spinner(t("processing")):
                    ok = safe_action(lambda: s.pipeline.listener_message(message))
                if ok:
                    changed(store)
                    st.rerun()


def preparation():
    s = st.session_state
    result = s.pipeline.state()["guidance_result"]
    title_column, download_column = st.columns([3, 2], vertical_alignment="center")
    title_column.header(t("preparation"))
    download_column.download_button(t("download"), result["lawyer_ready_summary"], file_name="lawyer-ready-summary.txt", mime="text/plain", icon=":material/download:", type="primary", width="stretch")
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.badge(t("complete_badge"), icon=":material/task_alt:", color="green")
        for item in result["past_cases_used"]:
            st.badge(item["id"], icon=":material/source:", color="gray")
    tabs = st.tabs(t("tabs").split("|"))
    with tabs[0]:
        with st.container(key="preparation-summary", gap="xsmall"):
            st.subheader(t("summary"), icon=":material/article:")
            st.text(result["situation_summary"])
        overview_left, overview_right = st.columns([3, 2], gap="medium")
        with overview_left:
            if result["options"]:
                st.subheader(t("options_note"), icon=":material/route:")
                for index, item in enumerate(result["options"], 1):
                    st.text(f"{index:02d}  {item['option']}")
        with overview_right:
            st.subheader(t("observations"), icon=":material/info:")
            st.caption(result["outcome_and_time"])
            for flag in result["red_flags"]:
                st.caption(":material/warning: " + flag)
    with tabs[1]:
        st.subheader(t("tabs").split("|")[1], icon=":material/folder_open:")
        for item in result["document_checklist"]:
            document_column, status_column = st.columns([3, 1], vertical_alignment="center")
            document_column.text(item["item"])
            status_column.badge(t(item["status"]), icon=":material/description:", color="green" if item["status"] == "mentioned" else "gray")
    with tabs[2]:
        st.subheader(t("tabs").split("|")[2], icon=":material/source:")
        for item in result["past_cases_used"]:
            with st.container(gap="xxsmall"):
                with st.container(horizontal=True, gap="small", vertical_alignment="center"):
                    st.badge(item["id"], color="green")
                    st.caption(t(item["source"]))
                st.text(item["why_similar"])
        for text in result["similar_cases"]:
            st.write(text)
    with tabs[3]:
        st.caption(t("options_note"))
        option_columns = st.columns(2)
        for index, item in enumerate(result["options"]):
            with option_columns[index % 2], st.container(border=True, gap="xsmall"):
                st.subheader(item["option"], icon=":material/route:")
                st.caption(t("benefit"))
                st.text(item["possible_benefit"])
                st.caption(t("tradeoff"))
                st.text(item["possible_tradeoff"])
    with tabs[4]:
        st.subheader(t("tabs").split("|")[4], icon=":material/forum:")
        question_columns = st.columns(2)
        for index, text in enumerate(result["questions_for_lawyer"]):
            question_columns[index % 2].text(f"{index + 1:02d}  {text}")
    with tabs[5]:
        st.caption(t("referrals"))
        for column, kind in zip(st.columns(2), ("legal_aid", "lawyers")):
            with column:
                st.subheader(t(kind))
                for item in result["referrals"][kind]:
                    with st.container(border=True, gap="xsmall"):
                        st.subheader(item["name"], icon=":material/support_agent:" if kind == "legal_aid" else ":material/work_outline:")
                        st.text(f"{item['city']} · {item.get('email', '')}")
                if not result["referrals"][kind]:
                    st.write(t("no_referrals"))
    st.caption(result["disclaimer"])
    if st.button(t("completion")):
        s.view = "completion"
        st.rerun()
    with st.expander(t("activity"), type="compact"):
        activity(s.pipeline.state(), show_history=False)
        st.caption(t("event_history"))
        activity_history(s.pipeline.state())


def completion():
    s = st.session_state
    st.header(t("completion"))
    st.write(t("resolved_question"))
    resolved = st.segmented_control(t("resolved_question"), ["not_yet", "resolved"], format_func=t, default=None if "learn_resolved" in s else "not_yet", required=True, key="learn_resolved", label_visibility="collapsed")
    if resolved == "not_yet":
        s.preview = None
        st.caption(t("private_note"))
        return
    private_column, share_column = st.columns(2)
    private_column.subheader(t("private"), icon=":material/lock:")
    private_column.caption(t("private_note"))
    share_column.subheader(t("share"), icon=":material/shield:")
    share_column.caption(t("share_note"))
    sharing = st.segmented_control(t("share"), ["private", "share"], format_func=t, default=None if "learn_sharing" in s else "private", required=True, key="learn_sharing", label_visibility="collapsed")
    if sharing == "private":
        s.preview = None
        st.caption(t("private_note"))
        return
    if s.get("shared"):
        st.success(t("thanks"))
        st.caption(s.shared)
        return
    case = s.pipeline.state()["case_file"]
    if case["legal_area"] != "land_dispute":
        st.caption(t("unsupported"))
        return
    st.write(t("share_note"))
    with st.form("learning", border=False):
        outcome_column, months_column, cost_column = st.columns(3)
        outcome = outcome_column.selectbox(t("outcome"), ["settled", "favourable", "unfavourable"], format_func=t)
        months = months_column.number_input(t("months"), min_value=0, value=0, step=1)
        cost = cost_column.selectbox(t("cost"), ["low", "medium", "high"], format_func=t)
        tried_column, factors_column = st.columns(2)
        tried = tried_column.text_area(t("tried"), height=80)
        factors = factors_column.text_area(t("factors"), height=80)
        lesson = st.text_area(t("lesson"), height=80)
        preview_action = st.form_submit_button(t("preview"), disabled=bool(s.get("preview")))
    if preview_action:
        def preview():
            # Profile name is an explicit anonymization hint, not a story fact.
            memory_case = {**case, "name": s.profile["display_name"]}
            details = {"outcome": outcome, "time_taken_months": int(months), "cost_level": cost,
                       "options_tried": [v.strip() for v in tried.splitlines() if v.strip()],
                       "key_factors": [v.strip() for v in factors.splitlines() if v.strip()], "lesson": lesson}
            s.preview = memory.prepare_memory_record(memory_case, details, consent=True)
            s.pop("learn_final_consent", None)
        with st.spinner(t("processing")):
            safe_action(preview)
    if s.get("preview"):
        with st.container(border=True):
            st.subheader(t("exact"), icon=":material/visibility:")
            st.json(s.preview)
        consent = st.checkbox(t("consent"), key="learn_final_consent")
        if st.button(t("confirm_share"), disabled=not consent):
            def confirm():
                saved = memory.confirm_save_memory(s.preview, confirmed=True)
                s.shared = saved["id"]
                s.preview = None
            if safe_action(confirm):
                st.rerun()


def main():
    st.set_page_config(page_title="Counsel Before Court", page_icon=":material/gavel:", layout="wide")
    st.html(CSS)
    s = st.session_state
    s.setdefault("view", "welcome")
    s.setdefault("dirty", False)
    brand_column, controls_column = st.columns([4, 1], vertical_alignment="center")
    with brand_column.container(horizontal=True, vertical_alignment="center", gap="xsmall"):
        st.image(LOGO, width=44)
        st.html(header(t("guide")), width="content")
    with controls_column:
        st.segmented_control("English | मराठी", ["English", "मराठी"], default=None if "language" in s else "English", required=True, key="language", label_visibility="collapsed", width="stretch")
        if s.get("profile"):
            st.caption(s.profile["display_name"])
    store = PrivateHistory()
    if s.get("profile"):
        if controls_column.button(t("cases"), width="stretch", icon=":material/folder_open:"):
            s.view = "cases"
            st.rerun()
        if s.get("dirty"):
            st.warning(t("save_error"))
            if st.button(t("retry_save")):
                if safe_action(lambda: persist(store)):
                    st.rerun()
        if s.view != "cases" and s.get("pipeline"):
            stage = s.pipeline.state()["current_stage"]
            if s.view != "completion":
                s.view = workflow_view(s.pipeline.state())
            elif stage != "COMPLETE":
                s.view = workflow_view(s.pipeline.state())
    else:
        s.view = "welcome"
    views = {"welcome": lambda: welcome(store), "cases": lambda: my_cases(store),
             "workspace": lambda: workspace(store), "evidence": lambda: workspace(store, True),
             "preparation": preparation, "completion": completion}
    safe_action(views[s.view])


if __name__ == "__main__":
    main()
