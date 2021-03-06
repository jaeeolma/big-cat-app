var el = x => document.getElementById(x);

function showPicker(inputId) { el('file-input').click(); }

function showPicked(input) {
    el('upload-label').innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function (e) {
        el('image-picked').src = e.target.result;
        el('image-picked').className = '';
    }
    reader.readAsDataURL(input.files[0]);
}

function analyze() {
    var uploadFiles = el('file-input').files;
    if (uploadFiles.length != 1) alert('Please select 1 file to analyze!');

    el('analyze-button').innerHTML = 'Analyzing...';
    var xhr = new XMLHttpRequest();
    var loc = window.location
    xhr.open('POST', `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`, true);
    xhr.onerror = function () { el('analyze-button').innerHTML = 'Analyze'; alert(xhr.responseText);}
    xhr.onload = function (e) {
        if (this.readyState === 4 && this.status === 200) {
            var response = JSON.parse(e.target.responseText);
            el('best-result-label').innerHTML = `This wild cat is ${response['best_result']}`;
            el('best-confidence-label').innerHTML = `with ${response['best_confidence']} confidence.`;
            el('second-result-label').innerHTML = `Other possibilities are ${response['second_result']}`;
            el('second-confidence-label').innerHTML = `with ${response['second_confidence']} confidence`;
            el('third-result-label').innerHTML = `and ${response['third_result']}`;
            el('third-confidence-label').innerHTML = `with ${response['third_confidence']} confidence.`;
        }
        el('analyze-button').innerHTML = 'Analyze';
    }

    var fileData = new FormData();
    fileData.append('file', uploadFiles[0]);
    xhr.send(fileData);
}

function analyzeUrl() {
    var extUrl = el('url-input').value;

    el('url-button').innerHTML = 'Analyzing...';
    var xhr = new XMLHttpRequest();
    var loc = window.location
    xhr.open('GET', `${loc.protocol}//${loc.hostname}:${loc.port}/analyze_url?url=` + extUrl, true);
    xhr.onerror = function () { alert(xhr.responseText);}
    xhr.onload = function (e) {
        if (this.readyState === 4 && this.status === 200) {
            var response = JSON.parse(e.target.responseText);
            el('image-picked').src = extUrl;
            el('image-picked').className = '';
            el('upload-label').innerHTML = extUrl;
            el('best-result-label').innerHTML = `This wild cat is ${response['best_result']}`;
            el('best-confidence-label').innerHTML = `with ${response['best_confidence']} confidence.`;
            el('second-result-label').innerHTML = `Other possibilities are ${response['second_result']}`;
            el('second-confidence-label').innerHTML = `with ${response['second_confidence']} confidence`;
            el('third-result-label').innerHTML = `and ${response['third_result']}`;
            el('third-confidence-label').innerHTML = `with ${response['third_confidence']} confidence.`;
            el('file-input').value = "";
        } else if (this.readyState === 4 && this.status === 500) {
            alert(xhr.responseText);
        }
        
        el('url-button').innerHTML = 'Or fetch and analyze URL';
    }

    xhr.send();
}

