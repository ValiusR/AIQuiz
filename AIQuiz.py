import streamlit as st
import easyocr
import asyncio
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

asyncio.set_event_loop(asyncio.new_event_loop())

# teksto atpazinimo is vaizdo modelio ikrovimas
reader = easyocr.Reader(['lt'])

# vertimui reikalingi moduliai
translator_model = "Helsinki-NLP/opus-mt-tc-big-lt-en"  # lt to en
translator_model_back = "Helsinki-NLP/opus-mt-tc-big-en-lt"  # en to lt
translator_lt_en = pipeline("translation", model=translator_model)
translator_en_lt = pipeline("translation", model=translator_model_back)

# modelis naudojamas santrumpai sukurti
summarizer_model_name = "facebook/bart-large-cnn" 
try:
    summarizer = pipeline("summarization", model=summarizer_model_name)
except Exception as e:
    st.error(f"Klaida su santraukos modeliu: {e}")
    st.stop()

# Streamlit UI Setup
st.title("Lietuviško teksto iš nuotraukos santraukos sukūrimas naudojant DI")
st.write("Įkelkite nuotrauką su tekstu:")

# Upload image file
uploaded_file = st.file_uploader("Įkelkite nuotrauką...", type=["png", "jpg", "jpeg"])

def preprocess_text(text):
    text = text.replace("-\n", "").replace("- ", "")  
    text = re.sub(r"[^a-zA-ZąčęėįšųūžĄČĘĖĮŠŲŪŽ0-9\s\.,;:]", "", text)  
    return text

if uploaded_file:
    st.image(uploaded_file, caption="Įkelta nuotrauka", use_container_width=True)

    with st.spinner("Gaunamas tekstas..."):
        extracted_text = reader.readtext(uploaded_file.read(), detail=0)
        extracted_text = " ".join(extracted_text)

    if extracted_text:
        st.subheader("Gautas tekstas:")
        st.write(extracted_text)

        # Preprocess the extracted text
        processed_text = preprocess_text(extracted_text)

        st.subheader("Sutvarkytas tekstas:")
        st.write(processed_text)

        # Translate Lithuanian text → English
        with st.spinner("Tekstas verčiamas..."):
            translated_text = translator_lt_en(processed_text)[0]['translation_text']

        st.subheader("Tekstas, išverstas į anglų kalbą:")
        st.write(translated_text)

        # Generate Summary
        with st.spinner("Gaunama santrauka..."):
            try:
                summary_output = summarizer(translated_text, max_length=100, min_length=30, do_sample=False)
                summary_english = summary_output[0]['summary_text']

                st.subheader("Santrauka anglų kalba:")
                st.write(summary_english)

                # Translate Summary Back to Lithuanian
                with st.spinner("Verčiama santrauka..."):
                    summary_lithuanian = translator_en_lt(summary_english)[0]['translation_text']

                st.subheader("Santrauka Lietuviškai:")
                st.write(summary_lithuanian)

            except Exception as e:
                st.error(f"Klaida gaunant santrauką: {e}")
    else:
        st.warning("Nerasta jokio teksto, pabandykite iš naujo.")
