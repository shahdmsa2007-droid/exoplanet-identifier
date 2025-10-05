import streamlit as st, sys, platform
st.title("✅ Hello from Streamlit Cloud")
st.write("Python:", sys.version)
st.write("Platform:", platform.platform())
st.success("If you can see this, the runtime is healthy. We'll restore the full app next.")
