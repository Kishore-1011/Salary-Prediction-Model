const FIELDS = ["experience", "test_score", "interview_score"];

const form = document.getElementById("candidate-form");
const submitButton = document.getElementById("submit");
const formError = document.getElementById("form-error");
const emptyState = document.getElementById("empty-state");
const result = document.getElementById("result");
const warnings = document.getElementById("warnings");

const formatNumber = (value) => value.toLocaleString("en-US");

function setText(id, text) {
    document.getElementById(id).textContent = text;
}

function clearMessages() {
    formError.hidden = true;
    for (const field of FIELDS) {
        document.getElementById(`${field}-error`).hidden = true;
        document.getElementById(field).removeAttribute("aria-invalid");
    }
}

function showEmptyState() {
    result.hidden = true;
    emptyState.hidden = false;
}

function showFormError(message) {
    formError.textContent = message;
    formError.hidden = false;
    showEmptyState();
}

function showFieldErrors(errors) {
    let firstInvalid = null;
    for (const field of FIELDS) {
        if (!errors[field]) {
            continue;
        }
        const input = document.getElementById(field);
        const message = document.getElementById(`${field}-error`);
        message.textContent = errors[field];
        message.hidden = false;
        input.setAttribute("aria-invalid", "true");
        firstInvalid = firstInvalid || input;
    }
    if (errors.form) {
        showFormError(errors.form);
    }
    showEmptyState();
    if (firstInvalid) {
        firstInvalid.focus();
    }
}

function showResult(data) {
    setText("salary", formatNumber(data.predicted_salary));
    setText(
        "range",
        `Likely between ${formatNumber(data.low)} and ${formatNumber(data.high)}, based on a typical error of about ${formatNumber(data.typical_error)}.`
    );
    setText(
        "note",
        `This is a rough estimate from ${data.training_rows} training records. Don't use it on its own to set pay.`
    );

    warnings.replaceChildren(
        ...data.warnings.map((text) => {
            const item = document.createElement("li");
            item.textContent = text;
            return item;
        })
    );
    warnings.hidden = data.warnings.length === 0;

    emptyState.hidden = true;
    result.hidden = false;
}

function readForm() {
    const candidate = {};
    for (const field of FIELDS) {
        const value = document.getElementById(field).value;
        candidate[field] = value === "" ? null : Number(value);
    }
    return candidate;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearMessages();
    submitButton.disabled = true;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(readForm()),
        });

        if (response.status === 422) {
            showFieldErrors((await response.json()).errors);
        } else if (response.ok) {
            showResult(await response.json());
        } else {
            showFormError("Something went wrong on the server. Please try again.");
        }
    } catch {
        showFormError("Couldn't reach the server. Check that the app is still running.");
    } finally {
        submitButton.disabled = false;
    }
});
