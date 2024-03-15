// Function to simulate a loading delay
function simulateLoading() {
    return new Promise(resolve => {
        setTimeout(resolve, 3000); // Adjust the delay time as needed (3 seconds in this example)
    });
}

// Function to show the smartphone and hide the loading element
function showSmartphone() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('smartphone').style.display = 'block';
}

// Function to submit text to Python (as in previous examples)
// Function to submit text to Python (as in previous examples)
function submitText() {
    var inputText = document.getElementById("inputText").value;
    sendTextToPython(inputText);

    // Get the result element
    var resultElement = document.getElementById('result');

    // Show the result element with fadeInOut animation
    resultElement.style.opacity = 1;
    resultElement.style.transition = 'opacity 1s ease-in-out';

    // Remove the opacity after the animation duration to reset for the next submit
    setTimeout(function() {
        resultElement.style.opacity = 0;
    }, 10000); // Adjust the timeout to match your animation duration
}


// Function to start speech recognition (as in previous examples)
function startListening() {
    var recognition = new webkitSpeechRecognition() || new SpeechRecognition();
    recognition.lang = 'en-US';

    recognition.onresult = function(event) {
        var transcript = event.results[0][0].transcript;
        document.getElementById("inputText").value = transcript;
        sendTextToPython(transcript);
    };

    recognition.start();
}

// Function to send text to Python (as in previous examples)
function sendTextToPython(text) {
    // Send the input text to Python using an API endpoint
//     fetch('/process_text', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ text: text }),
//     })
//     .then(response => response.json())
//     .then(data => {
//         document.getElementById("result").innerText = "Result : " + data.result;
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// }
// // Simulate loading when the page loads
// simulateLoading().then(() => {
//     showSmartphone();
// });
fetch('/process_text', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: text }),
})
.then(response => response.json())
.then(data => {
    // Display the result on the smartphone screen
    var smartphoneResultElement = document.getElementById('smartphone-result');
    smartphoneResultElement.innerText = data.result;
})
.catch(error => {
    console.error('Error:', error);
});
}
simulateLoading().then(() => {
    showSmartphone();
});