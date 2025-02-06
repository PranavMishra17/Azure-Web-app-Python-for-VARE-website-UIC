// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license.

// Global objects
var avatarSynthesizer;
var peerConnection;
var useTcpForWebRTC = false;
var previousAnimationFrameTimestamp = 0;

// Hardcoded configurations
const CONFIG = {
    cogSvcRegion: 'westus2',
    cogSvcSubKey: 'c897d534a33b4dd7a31e73026200226b',
    privateEndpointEnabled: false,
    privateEndpoint: '',
    backgroundColor: '#FFFFFFFF',
    videoCrop: true
};

// Remove or replace the log function with console.log
const log = msg => {
    console.log(msg);
};

// Setup WebRTC
function setupWebRTC(iceServerUrl, iceServerUsername, iceServerCredential) {
    // Create WebRTC peer connection
    peerConnection = new RTCPeerConnection({
        iceServers: [{
            urls: [ useTcpForWebRTC ? iceServerUrl.replace(':3478', ':443?transport=tcp') : iceServerUrl ],
            username: iceServerUsername,
            credential: iceServerCredential
        }],
        iceTransportPolicy: useTcpForWebRTC ? 'relay' : 'all'
    })

    // Fetch WebRTC video stream and mount it to an HTML video element
    peerConnection.ontrack = function (event) {
        // Clean up existing video element if there is any
        remoteVideoDiv = document.getElementById('remoteVideo')
        for (var i = 0; i < remoteVideoDiv.childNodes.length; i++) {
            if (remoteVideoDiv.childNodes[i].localName === event.track.kind) {
                remoteVideoDiv.removeChild(remoteVideoDiv.childNodes[i])
            }
        }

        const mediaPlayer = document.createElement(event.track.kind)
        mediaPlayer.id = event.track.kind
        mediaPlayer.srcObject = event.streams[0]
        mediaPlayer.autoplay = true
        document.getElementById('remoteVideo').appendChild(mediaPlayer)
        document.getElementById('videoLabel').hidden = true
        document.getElementById('overlayArea').hidden = false

        if (event.track.kind === 'video') {
            mediaPlayer.playsInline = true
            remoteVideoDiv = document.getElementById('remoteVideo')
            canvas = document.getElementById('canvas')
            if (document.getElementById('transparentBackground').checked) {
                remoteVideoDiv.style.width = '0.1px'
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
                canvas.hidden = false
            } else {
                canvas.hidden = true
            }

            mediaPlayer.addEventListener('play', () => {
                if (document.getElementById('transparentBackground').checked) {
                    window.requestAnimationFrame(makeBackgroundTransparent)
                } else {
                    remoteVideoDiv.style.width = mediaPlayer.videoWidth / 2 + 'px'
                }
            })
        }
        else
        {
            // Mute the audio player to make sure it can auto play, will unmute it when speaking
            // Refer to https://developer.mozilla.org/en-US/docs/Web/Media/Autoplay_guide
            mediaPlayer.muted = true
        }
    }

    // Listen to data channel, to get the event from the server
    peerConnection.addEventListener("datachannel", event => {
        const dataChannel = event.channel
        dataChannel.onmessage = e => {
            console.log("[" + (new Date()).toISOString() + "] WebRTC event received: " + e.data)
        }
    })

    // This is a workaround to make sure the data channel listening is working by creating a data channel from the client side
    c = peerConnection.createDataChannel("eventChannel")

    peerConnection.oniceconnectionstatechange = e => {
        console.log("WebRTC status: " + peerConnection.iceConnectionState);
    
        const speakButton = document.getElementById('speak');
        
        if (peerConnection.iceConnectionState === 'connected') {
            if (speakButton) speakButton.disabled = false;
        }
    
        if (peerConnection.iceConnectionState === 'disconnected' || peerConnection.iceConnectionState === 'failed') {
            if (speakButton) speakButton.disabled = true;
        }
    };

    // Offer to receive 1 audio, and 1 video track
    peerConnection.addTransceiver('video', { direction: 'sendrecv' })
    peerConnection.addTransceiver('audio', { direction: 'sendrecv' })

// start avatar, establish WebRTC connection
avatarSynthesizer.startAvatarAsync(peerConnection).then((r) => {
    if (r.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
        console.log("[" + (new Date()).toISOString() + "] Avatar started. Result ID: " + r.resultId)
    } else {
        console.log("[" + (new Date()).toISOString() + "] Unable to start avatar. Result ID: " + r.resultId)
        if (r.reason === SpeechSDK.ResultReason.Canceled) {
            let cancellationDetails = SpeechSDK.CancellationDetails.fromResult(r)
            if (cancellationDetails.reason === SpeechSDK.CancellationReason.Error) {
                console.log(cancellationDetails.errorDetails)
            }
        }
    }
}).catch((error) => {
    console.log("[" + (new Date()).toISOString() + "] Avatar failed to start. Error: " + error)
});
}

// Make video background transparent by matting
function makeBackgroundTransparent(timestamp) {
    // Throttle the frame rate to 30 FPS to reduce CPU usage
    if (timestamp - previousAnimationFrameTimestamp > 30) {
        video = document.getElementById('video')
        tmpCanvas = document.getElementById('tmpCanvas')
        tmpCanvasContext = tmpCanvas.getContext('2d', { willReadFrequently: true })
        tmpCanvasContext.drawImage(video, 0, 0, video.videoWidth, video.videoHeight)
        if (video.videoWidth > 0) {
            let frame = tmpCanvasContext.getImageData(0, 0, video.videoWidth, video.videoHeight)
            for (let i = 0; i < frame.data.length / 4; i++) {
                let r = frame.data[i * 4 + 0]
                let g = frame.data[i * 4 + 1]
                let b = frame.data[i * 4 + 2]
                if (g - 150 > r + b) {
                    // Set alpha to 0 for pixels that are close to green
                    frame.data[i * 4 + 3] = 0
                } else if (g + g > r + b) {
                    // Reduce green part of the green pixels to avoid green edge issue
                    adjustment = (g - (r + b) / 2) / 3
                    r += adjustment
                    g -= adjustment * 2
                    b += adjustment
                    frame.data[i * 4 + 0] = r
                    frame.data[i * 4 + 1] = g
                    frame.data[i * 4 + 2] = b
                    // Reduce alpha part for green pixels to make the edge smoother
                    a = Math.max(0, 255 - adjustment * 4)
                    frame.data[i * 4 + 3] = a
                }
            }

            canvas = document.getElementById('canvas')
            canvasContext = canvas.getContext('2d')
            canvasContext.putImageData(frame, 0, 0);
        }

        previousAnimationFrameTimestamp = timestamp
    }

    window.requestAnimationFrame(makeBackgroundTransparent)
}
// Do HTML encoding on given text
function htmlEncode(text) {
    const entityMap = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '/': '&#x2F;'
    };

    return String(text).replace(/[&<>"'\/]/g, (match) => entityMap[match])
}


window.startSession = () => {
    // Get URL parameters
    const urlParams = new URLSearchParams(window.location.search);

    const avatarMap = {
        'max-business': 'Max-business',
        'lisa-casual': 'Lisa-casual-sitting',
        'dr-david-avenetti': 'drdavid-professional',
        'prof-zalake': 'professor-business'
    };
    
    const mappedAvatar = urlParams.get('version');
    const avatarVersion = avatarMap[mappedAvatar] || 'Harry-business'; 

    console.log(`Using avatar: ${avatarVersion}`);

    let speechSynthesisConfig = SpeechSDK.SpeechConfig.fromSubscription(
        CONFIG.cogSvcSubKey, 
        CONFIG.cogSvcRegion
    );

    const videoFormat = new SpeechSDK.AvatarVideoFormat();
    if (CONFIG.videoCrop) {
        videoFormat.setCropRange(
            new SpeechSDK.Coordinate(600, 0),
            new SpeechSDK.Coordinate(1320, 1080)
        );
    }

    const avatarConfig = new SpeechSDK.AvatarConfig(
        avatarVersion,  // from URL
        'business',     // hardcoded style
        videoFormat
    );
    avatarConfig.customized = false;
    avatarConfig.backgroundColor = CONFIG.backgroundColor;

    const xhr = new XMLHttpRequest();
    xhr.open("GET", `https://${CONFIG.cogSvcRegion}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1`);
    xhr.setRequestHeader("Ocp-Apim-Subscription-Key", CONFIG.cogSvcSubKey);
    xhr.addEventListener("readystatechange", function() {
        if (this.readyState === 4) {
            if (this.status === 200) {
                const responseData = JSON.parse(this.responseText);
                const iceServerUrl = responseData.Urls[0];
                const iceServerUsername = responseData.Username;
                const iceServerCredential = responseData.Password;

                avatarSynthesizer = new SpeechSDK.AvatarSynthesizer(speechSynthesisConfig, avatarConfig);
                avatarSynthesizer.avatarEventReceived = function (s, e) {
                    console.log("[Event]", e.description);
                }

                setupWebRTC(iceServerUrl, iceServerUsername, iceServerCredential);
            } else {
                console.error("Failed to get token:", this.status, this.statusText);
            }
        }
    });
    xhr.send();
};

window.speak = () => {
    const speakButton = document.getElementById('speak');
    const spokenText = document.getElementById('spokenText').value;
    if (!spokenText) return;

    if (speakButton) speakButton.disabled = true;
    const voiceMap = {
        'max-business': 'en-US-JacobNeural',
        'lisa-casual': 'en-US-NancyNeural',
        'dr-david-avenetti': 'en-US-DavidNeural',
        'prof-zalake': 'en-US-JacobNeural'
    };

    const urlParams = new URLSearchParams(window.location.search);
    const avatarVersion = urlParams.get('version');
    const ttsVoice = voiceMap[avatarVersion] || 'en-US-JacobNeural';

    let spokenSsml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
        <voice name='${ttsVoice}'>
            ${htmlEncode(spokenText)}
        </voice>
    </speak>`;

    avatarSynthesizer.speakSsmlAsync(spokenSsml).then((result) => {
        if (speakButton) speakButton.disabled = false;
        if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
            console.log("Speech synthesized successfully");
        } else {
            console.log("Speech synthesis failed");
        }
    }).catch(console.error);
};
