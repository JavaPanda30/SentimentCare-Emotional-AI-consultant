document.addEventListener("DOMContentLoaded", () => {
  const ReactUserInput = document.getElementById("userInput");
  const ReactGetAdviceButton = document.getElementById("getAdviceButton");
  const ReactResponseContainer = document.getElementById("responseContainer");
  const ReactResponseText = document.getElementById("responseText");
  const ReactCloseResponseButton = document.getElementById("closeResponseButton");

  ReactGetAdviceButton.addEventListener("click", async () => {
    const input = ReactUserInput.value.trim();

    if (!input) {
      alert("Please type something to get advice.");
      return;
    }

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ userInput: input }),
      });

      const result = await response.json();

      if (result.success) {
        const { detectedEmotion, adviceResponse } = result;
        const advicePoints = adviceResponse.split(/\n/).filter((line) => line);

        // Update the response box
        ReactResponseText.innerHTML = `
                  <strong>Emotion Detected: </strong>${detectedEmotion}
                  <br/>
                  <strong>Advice:</strong>
                  <ul>
                      ${advicePoints
                        .map((point) => `<li>${point}</li>`)
                        .join("")}
                  </ul>`;
      } else {
        ReactResponseText.innerText =
          "Sorry, unable to fetch detailed advice at the moment.";
      }

      ReactResponseContainer.classList.remove("hidden");
    } catch (error) {
      console.error("Error:", error);
      ReactResponseText.innerText =
        "Sorry, there was an error. Please try again later.";
      ReactResponseContainer.classList.remove("hidden");
    }

    ReactUserInput.value = "";
  });

  ReactCloseResponseButton.addEventListener("click", () => {
    ReactResponseContainer.classList.add("hidden");
  });
});
