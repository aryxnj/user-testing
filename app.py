import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

def scroll_to_top():
    components.html(
        """
        <script>
        window.scrollTo({ top: 0, behavior: 'smooth' });
        </script>
        """,
        height=1,  # Minimal height to ensure the component renders
        width=1,   # Minimal width to ensure the component renders
        key=f"scroll_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"  # Unique key without dots
    )

# Call scroll_to_top
scroll_to_top()

st.write("If you see this, the scroll_to_top function executed without errors.")
