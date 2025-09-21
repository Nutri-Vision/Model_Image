document.addEventListener('DOMContentLoaded', function() {
    console.log("Food Classifier JS loaded");

    // Get elements
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const mealList = document.getElementById('mealList');
    const emptyState = document.getElementById('emptyState');

    let currentFile = null;

    // File input handler
    fileInput.addEventListener('change', function(e) {
        if (this.files && this.files.length > 0) {
            handleFileSelect(this.files[0]);
        }
    });

    // Upload area click handler
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // Handle file selection (only show name, not preview)
    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }
        currentFile = file;
        analyzeBtn.disabled = false;

        uploadArea.innerHTML = `
            <p><strong>Selected File:</strong> ${file.name}</p>
            <button class="btn btn-secondary" onclick="clearFileSelection()">Change Image</button>
        `;
    }

    // Clear file selection
    window.clearFileSelection = function() {
        currentFile = null;
        analyzeBtn.disabled = true;
        fileInput.value = '';
        uploadArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt" style="font-size: 48px; margin-bottom: 15px;"></i>
            <p>Drag & drop your food image here or click to browse</p>
            <button class="btn btn-primary">Select Image</button>
        `;
    };

    // Analyze button
    analyzeBtn.addEventListener('click', function() {
        if (!currentFile) {
            alert('Please select an image first');
            return;
        }

        const quantity = document.getElementById('quantity').value || '100';
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('quantity', quantity);

        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        analyzeBtn.disabled = true;

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayResults(data);
                loadMealHistory();
            } else {
                throw new Error(data.error || 'Unknown error');
            }
        })
        .catch(error => {
            console.error(error);
            alert('Error: ' + error.message);
        })
        .finally(() => {
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Food';
            analyzeBtn.disabled = false;
        });
    });

    // Display results
    function displayResults(data) {
        document.getElementById('foodName').textContent = data.prediction;
        document.getElementById('confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById('quantityDisplay').textContent = document.getElementById('quantity').value;

        document.getElementById('calories').textContent = data.nutrition.calories + ' kcal';
        document.getElementById('protein').textContent = data.nutrition.protein + 'g';
        document.getElementById('carbs').textContent = data.nutrition.carbs + 'g';
        document.getElementById('fat').textContent = data.nutrition.fat + 'g';
        document.getElementById('fiber').textContent = data.nutrition.fiber + 'g';

        resultsSection.style.display = 'block';
    }

    // Load meal history
    function loadMealHistory() {
        fetch('/meals')
        .then(response => response.json())
        .then(meals => {
            renderMealHistory(meals);
            updateTotals(meals);
        })
        .catch(err => console.error("Error loading meals:", err));
    }

    // Render meals
    function renderMealHistory(meals) {
        if (!meals || meals.length === 0) {
            emptyState.style.display = 'block';
            mealList.innerHTML = '';
            return;
        }

        emptyState.style.display = 'none';
        mealList.innerHTML = '';

        meals.forEach((meal, index) => {
            const mealItem = document.createElement('div');
            mealItem.className = 'meal-item';
            const mealDate = new Date(meal.timestamp);

            mealItem.innerHTML = `
                <div class="meal-details">
                    <h4>${meal.food}</h4>
                    <p>${mealDate.toLocaleDateString()} ${mealDate.toLocaleTimeString()}</p>
                    <p>Quantity: ${meal.quantity}g</p>
                </div>
                <div class="meal-nutrition">
                    <div class="calories">${meal.nutrition.calories} kcal</div>
                    <div class="details">P: ${meal.nutrition.protein}g | C: ${meal.nutrition.carbs}g | F: ${meal.nutrition.fat}g</div>
                </div>
                <button class="delete-meal" data-meal-id="${index}">
                    <i class="fas fa-trash"></i>
                </button>
            `;
            mealList.appendChild(mealItem);
        });

        document.querySelectorAll('.delete-meal').forEach(button => {
            button.addEventListener('click', function() {
                const mealId = this.getAttribute('data-meal-id');
                deleteMeal(mealId);
            });
        });
    }

    // Delete meal
    function deleteMeal(mealId) {
        fetch(`/meal/${mealId}`, { method: 'DELETE' })
        .then(response => response.json())
        .then(data => {
            if (data.success) loadMealHistory();
            else alert('Error deleting meal');
        })
        .catch(err => console.error("Delete error:", err));
    }

    // Update totals for today
    function updateTotals(meals) {
        const totals = { calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0 };
        const today = new Date().toDateString();

        meals.forEach(meal => {
            const mealDate = new Date(meal.timestamp).toDateString();
            if (mealDate === today) {
                totals.calories += meal.nutrition.calories;
                totals.protein  += meal.nutrition.protein;
                totals.carbs    += meal.nutrition.carbs;
                totals.fat      += meal.nutrition.fat;
                totals.fiber    += meal.nutrition.fiber;
            }
        });

        document.getElementById('totalCalories').textContent = totals.calories + ' kcal';
        document.getElementById('totalProtein').textContent  = totals.protein + ' g';
        document.getElementById('totalCarbs').textContent    = totals.carbs + ' g';
        document.getElementById('totalFat').textContent      = totals.fat + ' g';
        document.getElementById('totalFiber').textContent    = totals.fiber + ' g';
    }

    // Initial load
    loadMealHistory();
});
