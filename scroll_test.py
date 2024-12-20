import streamlit as st

# 1. Add a hidden anchor at the very top of the page
st.markdown('<a id="top"></a>', unsafe_allow_html=True)

# 2. Place the title after the anchor
st.title("Scroll to Top Test")

# 3. Add content to make the page scrollable
for i in range(50):
    st.write(f"**Line {i + 1}:** This is some sample content to make the page scrollable.")

# 4. Style for the scroll button
st.markdown(
    """
    <style>
    .scroll-button {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border: none;
        border-radius: 50%;
        text-align: center;
        text-decoration: none;
        font-size: 18px;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: background-color 0.3s;
        z-index: 1000; /* Ensure the button stays on top */
    }

    .scroll-button:hover {
        background-color: #45a049;
    }
    </style>
    <a href="#top" class="scroll-button" title="Scroll to Top">↑</a>
    """,
    unsafe_allow_html=True
)
