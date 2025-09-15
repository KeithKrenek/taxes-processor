/**
 * Financial Transaction Processing System - JavaScript Implementation
 * ================================================================
 * 
 * A modern, browser-based financial data processing system demonstrating
 * advanced JavaScript patterns, async programming, and web technologies.
 * 
 * Author: Portfolio Demonstration
 * Target Roles: AI Researcher, Applied Physicist, Technical Lead, Engineering Manager
 */

import * as XLSX from 'xlsx';
import _ from 'lodash';

/**
 * Configuration class for processing parameters
 */
class ProcessingConfig {
    constructor(options = {}) {
        // Categorization settings
        this.mlConfidenceThreshold = options.mlConfidenceThreshold || 0.8;
        this.enableFuzzyMatching = options.enableFuzzyMatching !== false;
        this.fuzzyThreshold = options.fuzzyThreshold || 85;
        
        // Duplicate detection settings
        this.temporalWindowDays = options.temporalWindowDays || 5;
        this.amountTolerance = options.amountTolerance || 0.01;
        this.similarityThreshold = options.similarityThreshold || 0.85;
        
        // Processing settings
        this.enableEnsemble = options.enableEnsemble !== false;
        this.randomSeed = options.randomSeed || 42;
        
        // Visualization settings
        this.chartTheme = options.chartTheme || 'modern';
        this.exportFormats = options.exportFormats || ['html', 'json'];
    }
    
    static fromJSON(configData) {
        return new ProcessingConfig(configData);
    }
}

/**
 * Transaction record class with validation and methods
 */
class TransactionRecord {
    constructor({
        date,
        amount,
        description,
        account,
        category = null,
        confidence = 0.0,
        isDuplicate = false,
        duplicateGroup = null,
        sourceSheet = '',
        rawData = {}
    }) {
        this.date = new Date(date);
        this.amount = parseFloat(amount);
        this.description = String(description || '');
        this.account = String(account || '');
        this.category = category;
        this.confidence = parseFloat(confidence);
        this.isDuplicate = Boolean(isDuplicate);
        this.duplicateGroup = duplicateGroup;
        this.sourceSheet = String(sourceSheet);
        this.rawData = rawData;
        
        this._validate();
    }
    
    _validate() {
        if (isNaN(this.date.getTime())) {
            throw new Error('Invalid date provided');
        }
        if (isNaN(this.amount)) {
            throw new Error('Invalid amount provided');
        }
        if (this.confidence < 0 || this.confidence > 1) {
            throw new Error('Confidence must be between 0 and 1');
        }
    }
    
    toJSON() {
        return {
            date: this.date.toISOString(),
            amount: this.amount,
            description: this.description,
            account: this.account,
            category: this.category,
            confidence: this.confidence,
            isDuplicate: this.isDuplicate,
            duplicateGroup: this.duplicateGroup,
            sourceSheet: this.sourceSheet,
            rawData: this.rawData
        };
    }
}

/**
 * Advanced Excel data loader with validation and preprocessing
 */
class DataLoader {
    constructor(config) {
        this.config = config;
        this.logger = console;
    }
    
    async loadWorkbook(file) {
        this.logger.info('Loading Excel workbook...');
        
        try {
            const arrayBuffer = await this._fileToArrayBuffer(file);
            const workbook = XLSX.read(arrayBuffer, {
                cellStyles: true,
                cellFormulas: true,
                cellDates: true,
                cellNF: true,
                sheetStubs: true
            });
            
            const transactions = await this._processWorkbook(workbook);
            
            this.logger.info(`Loaded ${transactions.length} transactions from ${workbook.SheetNames.length} sheets`);
            return transactions;
            
        } catch (error) {
            this.logger.error('Error loading workbook:', error);
            throw new Error(`Failed to load workbook: ${error.message}`);
        }
    }
    
    async _fileToArrayBuffer(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.onerror = e => reject(new Error('Failed to read file'));
            reader.readAsArrayBuffer(file);
        });
    }
    
    async _processWorkbook(workbook) {
        const transactions = [];
        
        // Filter out category reference sheets
        const transactionSheets = workbook.SheetNames.filter(
            name => !name.toLowerCase().includes('categories')
        );
        
        for (const sheetName of transactionSheets) {
            const sheet = workbook.Sheets[sheetName];
            const sheetData = XLSX.utils.sheet_to_json(sheet, { defval: '' });
            
            const sheetTransactions = await this._processSheet(sheetData, sheetName);
            transactions.push(...sheetTransactions);
        }
        
        return transactions;
    }
    
    async _processSheet(data, sheetName) {
        const transactions = [];
        const columnMapping = this._mapColumns(Object.keys(data[0] || {}));
        
        for (const [index, row] of data.entries()) {
            try {
                const transaction = this._createTransactionRecord(row, columnMapping, sheetName);
                if (transaction) {
                    transactions.push(transaction);
                }
            } catch (error) {
                this.logger.warn(`Error processing row ${index} in sheet ${sheetName}:`, error.message);
            }
        }
        
        return transactions;
    }
    
    _mapColumns(columns) {
        const mapping = {};
        
        // Date field mapping with fuzzy matching
        const datePatterns = ['date', 'posting date', 'transaction date', 'trans date'];
        mapping.date = this._findColumnMatch(columns, datePatterns);
        
        // Amount field mapping
        const amountPatterns = ['amount', 'transaction amount', 'debit', 'credit'];
        mapping.amount = this._findColumnMatch(columns, amountPatterns);
        
        // Description field mapping
        const descPatterns = ['description', 'memo', 'details', 'transaction details'];
        mapping.description = this._findColumnMatch(columns, descPatterns);
        
        return mapping;
    }
    
    _findColumnMatch(columns, patterns) {
        let bestMatch = null;
        let bestScore = 0;
        
        for (const pattern of patterns) {
            for (const column of columns) {
                const score = this._calculateSimilarity(
                    pattern.toLowerCase(), 
                    column.toLowerCase()
                );
                if (score > bestScore && score > 0.7) {
                    bestScore = score;
                    bestMatch = column;
                }
            }
        }
        
        return bestMatch;
    }
    
    _calculateSimilarity(str1, str2) {
        // Simple Levenshtein distance-based similarity
        const distance = this._levenshteinDistance(str1, str2);
        const maxLength = Math.max(str1.length, str2.length);
        return maxLength === 0 ? 1 : 1 - (distance / maxLength);
    }
    
    _levenshteinDistance(str1, str2) {
        const matrix = Array(str2.length + 1).fill(null).map(() => 
            Array(str1.length + 1).fill(null)
        );
        
        for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
        for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;
        
        for (let j = 1; j <= str2.length; j++) {
            for (let i = 1; i <= str1.length; i++) {
                const substitutionCost = str1[i - 1] === str2[j - 1] ? 0 : 1;
                matrix[j][i] = Math.min(
                    matrix[j][i - 1] + 1,         // insertion
                    matrix[j - 1][i] + 1,         // deletion
                    matrix[j - 1][i - 1] + substitutionCost // substitution
                );
            }
        }
        
        return matrix[str2.length][str1.length];
    }
    
    _createTransactionRecord(row, mapping, sheetName) {
        try {
            // Extract and validate date
            const dateField = mapping.date;
            if (!dateField || !row[dateField]) return null;
            
            const date = new Date(row[dateField]);
            if (isNaN(date.getTime())) return null;
            
            // Extract and validate amount
            const amountField = mapping.amount;
            if (!amountField || row[amountField] === undefined) return null;
            
            const amount = parseFloat(row[amountField]);
            if (isNaN(amount)) return null;
            
            // Extract description
            const descField = mapping.description;
            const description = this._cleanDescription(
                row[descField] ? String(row[descField]) : ''
            );
            
            return new TransactionRecord({
                date,
                amount,
                description,
                account: sheetName,
                sourceSheet: sheetName,
                rawData: { ...row }
            });
            
        } catch (error) {
            this.logger.warn('Error creating transaction record:', error.message);
            return null;
        }
    }
    
    _cleanDescription(description) {
        if (!description) return '';
        
        return description
            .replace(/\s+/g, ' ')                    // Normalize whitespace
            .replace(/[^\x20-\x7E]/g, '')            // Remove non-printable chars
            .replace(/\b(purchase|payment|transfer)\b/gi, '') // Remove common words
            .trim();
    }
}

/**
 * Machine learning-inspired transaction categorization system
 */
class TransactionCategorizer {
    constructor(config) {
        this.config = config;
        this.logger = console;
        
        // Category keyword mappings
        this.categoryKeywords = this._initializeCategoryKeywords();
        
        // Feature extraction components
        this.vocabulary = new Map();
        this.trained = false;
        this.categoryModel = null;
    }
    
    _initializeCategoryKeywords() {
        return {
            'Food & Dining': [
                'restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'taco', 'doordash',
                'uber eats', 'grubhub', 'dining', 'food', 'kitchen', 'deli', 'bakery',
                'mcdonald', 'starbucks', 'subway', 'chipotle', 'panera'
            ],
            'Transportation': [
                'uber', 'lyft', 'taxi', 'gas', 'fuel', 'parking', 'metro', 'bus',
                'train', 'airline', 'flight', 'rental car', 'auto', 'vehicle',
                'shell', 'exxon', 'bp', 'chevron', 'speedway'
            ],
            'Shopping': [
                'amazon', 'target', 'walmart', 'costco', 'store', 'retail', 'shopping',
                'clothing', 'apparel', 'electronics', 'purchase', 'ebay', 'etsy'
            ],
            'Utilities': [
                'electric', 'water', 'gas', 'internet', 'phone', 'cable', 'utility',
                'bill', 'service', 'provider', 'connection', 'comcast', 'verizon'
            ],
            'Healthcare': [
                'medical', 'doctor', 'pharmacy', 'hospital', 'clinic', 'health',
                'dental', 'vision', 'prescription', 'copay', 'cvs', 'walgreens'
            ],
            'Entertainment': [
                'movie', 'theater', 'concert', 'streaming', 'netflix', 'spotify',
                'entertainment', 'game', 'recreation', 'hobby', 'disney', 'hulu'
            ],
            'Financial Services': [
                'bank', 'atm', 'fee', 'interest', 'loan', 'credit', 'investment',
                'financial', 'advisor', 'brokerage', 'wells fargo', 'chase'
            ],
            'Home & Garden': [
                'home depot', 'lowes', 'garden', 'hardware', 'improvement', 'repair',
                'maintenance', 'contractor', 'furniture', 'ikea', 'ace hardware'
            ],
            'Professional Services': [
                'legal', 'accounting', 'consulting', 'professional', 'service',
                'attorney', 'cpa', 'advisor', 'tax', 'audit'
            ],
            'Education': [
                'school', 'university', 'college', 'education', 'tuition', 'books',
                'course', 'training', 'certification', 'coursera', 'udemy'
            ]
        };
    }
    
    async categorizeTransactions(transactions) {
        this.logger.info('Categorizing transactions...');
        
        const categorized = [];
        
        for (const transaction of transactions) {
            const { category, confidence } = await this._categorizeTransaction(transaction);
            
            transaction.category = category;
            transaction.confidence = confidence;
            categorized.push(transaction);
        }
        
        this.logger.info(`Categorized ${categorized.length} transactions`);
        return categorized;
    }
    
    async _categorizeTransaction(transaction) {
        // Try rule-based categorization first
        const ruleBasedResult = this._ruleBasedCategorization(transaction);
        
        if (ruleBasedResult.confidence >= this.config.mlConfidenceThreshold) {
            return ruleBasedResult;
        }
        
        // Try ML-inspired categorization
        const mlResult = await this._mlCategorization(transaction);
        
        // Return the result with higher confidence
        return mlResult.confidence > ruleBasedResult.confidence ? mlResult : ruleBasedResult;
    }
    
    _ruleBasedCategorization(transaction) {
        const description = transaction.description.toLowerCase();
        const amount = Math.abs(transaction.amount);
        
        // Keyword-based matching with confidence scoring
        let bestCategory = 'Uncategorized';
        let bestConfidence = 0.1;
        
        for (const [category, keywords] of Object.entries(this.categoryKeywords)) {
            const matches = keywords.filter(keyword => 
                description.includes(keyword.toLowerCase())
            );
            
            if (matches.length > 0) {
                // Calculate confidence based on match quality and quantity
                const confidence = Math.min(0.95, 
                    0.6 + (matches.length * 0.1) + 
                    (matches.some(match => description.startsWith(match.toLowerCase())) ? 0.2 : 0)
                );
                
                if (confidence > bestConfidence) {
                    bestCategory = category;
                    bestConfidence = confidence;
                }
            }
        }
        
        // Amount-based rules for edge cases
        if (bestConfidence < 0.5) {
            if (amount > 1000) {
                return { category: 'Large Purchase', confidence: 0.6 };
            } else if (amount < 10 && transaction.amount < 0) {
                return { category: 'Small Expense', confidence: 0.7 };
            }
        }
        
        return { category: bestCategory, confidence: bestConfidence };
    }
    
    async _mlCategorization(transaction) {
        // Simplified ML-inspired approach using feature vectors
        const features = this._extractFeatures(transaction);
        
        // Use a simple scoring system based on feature weights
        const categoryScores = {};
        
        for (const category of Object.keys(this.categoryKeywords)) {
            categoryScores[category] = this._calculateCategoryScore(features, category);
        }
        
        // Find best category
        const bestCategory = Object.keys(categoryScores).reduce((a, b) => 
            categoryScores[a] > categoryScores[b] ? a : b
        );
        
        const confidence = Math.min(0.9, categoryScores[bestCategory]);
        
        return { 
            category: confidence > 0.3 ? bestCategory : 'Uncategorized', 
            confidence 
        };
    }
    
    _extractFeatures(transaction) {
        const description = transaction.description.toLowerCase();
        const amount = Math.abs(transaction.amount);
        const date = transaction.date;
        
        return {
            // Text features
            wordCount: description.split(' ').length,
            hasDigits: /\d/.test(description),
            hasCommonWords: ['the', 'and', 'or'].some(word => description.includes(word)),
            
            // Amount features
            amount: amount,
            amountLog: Math.log1p(amount),
            isRound: amount === Math.round(amount),
            isNegative: transaction.amount < 0,
            amountCategory: this._categorizeAmount(amount),
            
            // Temporal features
            dayOfWeek: date.getDay(),
            dayOfMonth: date.getDate(),
            month: date.getMonth() + 1,
            isWeekend: date.getDay() === 0 || date.getDay() === 6,
            isMonthEnd: date.getDate() > 25,
            
            // Description features
            description: description
        };
    }
    
    _categorizeAmount(amount) {
        if (amount < 10) return 1;
        if (amount < 50) return 2;
        if (amount < 200) return 3;
        if (amount < 1000) return 4;
        return 5;
    }
    
    _calculateCategoryScore(features, category) {
        const keywords = this.categoryKeywords[category];
        let score = 0.1; // Base score
        
        // Text matching score
        const textMatches = keywords.filter(keyword => 
            features.description.includes(keyword.toLowerCase())
        ).length;
        score += textMatches * 0.3;
        
        // Amount-based adjustments
        if (category === 'Food & Dining' && features.amountCategory <= 3) {
            score += 0.2;
        } else if (category === 'Transportation' && features.amountCategory <= 4) {
            score += 0.15;
        } else if (category === 'Utilities' && features.isMonthEnd) {
            score += 0.25;
        }
        
        // Temporal adjustments
        if (category === 'Entertainment' && features.isWeekend) {
            score += 0.1;
        }
        
        return Math.min(0.9, score);
    }
}

/**
 * Advanced duplicate detection using multiple similarity metrics
 */
class DuplicateDetector {
    constructor(config) {
        this.config = config;
        this.logger = console;
    }
    
    async detectDuplicates(transactions) {
        this.logger.info('Detecting duplicate transactions...');
        
        // Sort transactions by date for efficient processing
        const sortedTransactions = [...transactions].sort((a, b) => a.date - b.date);
        
        const duplicateGroups = [];
        const processedIndices = new Set();
        
        for (let i = 0; i < sortedTransactions.length; i++) {
            if (processedIndices.has(i)) continue;
            
            const transaction = sortedTransactions[i];
            const duplicates = await this._findDuplicateCandidates(
                transaction, 
                sortedTransactions.slice(i + 1),
                i + 1
            );
            
            if (duplicates.length > 0) {
                const groupId = duplicateGroups.length;
                duplicateGroups.push([i, ...duplicates]);
                
                // Mark all transactions in group as duplicates
                [i, ...duplicates].forEach(idx => {
                    sortedTransactions[idx].isDuplicate = true;
                    sortedTransactions[idx].duplicateGroup = groupId;
                    processedIndices.add(idx);
                });
            }
        }
        
        this.logger.info(
            `Found ${duplicateGroups.length} duplicate groups affecting ${processedIndices.size} transactions`
        );
        
        return sortedTransactions;
    }
    
    async _findDuplicateCandidates(target, candidates, startIdx) {
        const duplicates = [];
        
        for (let i = 0; i < candidates.length; i++) {
            const candidate = candidates[i];
            const actualIdx = startIdx + i;
            
            // Check temporal proximity
            const daysDiff = Math.abs(target.date - candidate.date) / (1000 * 60 * 60 * 24);
            if (daysDiff > this.config.temporalWindowDays) continue;
            
            // Check amount similarity
            if (Math.abs(target.amount - candidate.amount) > this.config.amountTolerance) continue;
            
            // Check description similarity
            const similarity = await this._calculateDescriptionSimilarity(
                target.description, 
                candidate.description
            );
            
            if (similarity >= this.config.similarityThreshold) {
                duplicates.push(actualIdx);
            }
        }
        
        return duplicates;
    }
    
    async _calculateDescriptionSimilarity(desc1, desc2) {
        // Normalize descriptions
        const norm1 = this._normalizeDescription(desc1);
        const norm2 = this._normalizeDescription(desc2);
        
        // Calculate multiple similarity metrics
        const exactMatch = norm1 === norm2 ? 1 : 0;
        const tokenSimilarity = this._calculateTokenSimilarity(norm1, norm2);
        const editDistance = this._calculateEditDistanceSimilarity(norm1, norm2);
        
        // Weighted combination
        const combinedScore = (exactMatch * 0.5) + (tokenSimilarity * 0.3) + (editDistance * 0.2);
        
        return combinedScore;
    }
    
    _normalizeDescription(description) {
        return description
            .toLowerCase()
            .replace(/\b\d{4,}\b/g, '')           // Remove long numbers
            .replace(/\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/g, '') // Remove dates
            .replace(/\b\d{1,2}:\d{2}(:\d{2})?\b/g, '')    // Remove times
            .replace(/#\d+/g, '')                 // Remove reference numbers
            .replace(/\b(pos|atm)\b/g, '')        // Remove transaction types
            .replace(/\s+/g, ' ')                 // Normalize whitespace
            .trim();
    }
    
    _calculateTokenSimilarity(str1, str2) {
        const tokens1 = new Set(str1.split(' ').filter(t => t.length > 2));
        const tokens2 = new Set(str2.split(' ').filter(t => t.length > 2));
        
        if (tokens1.size === 0 && tokens2.size === 0) return 1;
        if (tokens1.size === 0 || tokens2.size === 0) return 0;
        
        const intersection = new Set([...tokens1].filter(t => tokens2.has(t)));
        const union = new Set([...tokens1, ...tokens2]);
        
        return intersection.size / union.size;
    }
    
    _calculateEditDistanceSimilarity(str1, str2) {
        const distance = this._levenshteinDistance(str1, str2);
        const maxLength = Math.max(str1.length, str2.length);
        
        return maxLength === 0 ? 1 : 1 - (distance / maxLength);
    }
    
    _levenshteinDistance(str1, str2) {
        const matrix = Array(str2.length + 1).fill(null).map(() => 
            Array(str1.length + 1).fill(null)
        );
        
        for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
        for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;
        
        for (let j = 1; j <= str2.length; j++) {
            for (let i = 1; i <= str1.length; i++) {
                const substitutionCost = str1[i - 1] === str2[j - 1] ? 0 : 1;
                matrix[j][i] = Math.min(
                    matrix[j][i - 1] + 1,         // insertion
                    matrix[j - 1][i] + 1,         // deletion
                    matrix[j - 1][i - 1] + substitutionCost // substitution
                );
            }
        }
        
        return matrix[str2.length][str1.length];
    }
}

/**
 * Interactive visualization system for financial data
 */
class FinancialVisualizer {
    constructor(config) {
        this.config = config;
        this.logger = console;
    }
    
    async createComprehensiveDashboard(transactions, containerId = 'dashboard') {
        this.logger.info('Creating comprehensive financial dashboard...');
        
        const container = document.getElementById(containerId);
        if (!container) {
            throw new Error(`Container element with id '${containerId}' not found`);
        }
        
        // Create dashboard structure
        container.innerHTML = this._createDashboardHTML();
        
        // Generate individual visualizations
        await Promise.all([
            this._createSpendingTrends(transactions, 'spending-trends'),
            this._createCategoryDistribution(transactions, 'category-distribution'),
            this._createAmountDistribution(transactions, 'amount-distribution'),
            this._createDuplicateAnalysis(transactions, 'duplicate-analysis'),
            this._createConfidenceAnalysis(transactions, 'confidence-analysis'),
            this._createTimelineView(transactions, 'timeline-view')
        ]);
        
        // Add summary statistics
        this._updateSummaryStats(transactions);
        
        this.logger.info('Dashboard created successfully');
    }
    
    _createDashboardHTML() {
        return `
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>Financial Transaction Analysis Dashboard</h1>
                    <div id="summary-stats" class="summary-stats"></div>
                </div>
                
                <div class="dashboard-grid">
                    <div class="chart-container">
                        <h3>Monthly Spending Trends</h3>
                        <div id="spending-trends"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Category Distribution</h3>
                        <div id="category-distribution"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Transaction Amount Distribution</h3>
                        <div id="amount-distribution"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Duplicate Detection Analysis</h3>
                        <div id="duplicate-analysis"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Categorization Confidence</h3>
                        <div id="confidence-analysis"></div>
                    </div>
                    
                    <div class="chart-container full-width">
                        <h3>Transaction Timeline</h3>
                        <div id="timeline-view"></div>
                    </div>
                </div>
            </div>
            
            <style>
                .dashboard-container {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }
                
                .dashboard-header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                
                .dashboard-header h1 {
                    color: #333;
                    margin-bottom: 20px;
                }
                
                .summary-stats {
                    display: flex;
                    justify-content: center;
                    gap: 30px;
                    flex-wrap: wrap;
                }
                
                .stat-item {
                    background: #f8f9fa;
                    padding: 15px 25px;
                    border-radius: 8px;
                    text-align: center;
                    min-width: 120px;
                }
                
                .stat-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }
                
                .stat-label {
                    font-size: 12px;
                    color: #666;
                    margin-top: 5px;
                }
                
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                }
                
                .chart-container {
                    background: white;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .chart-container.full-width {
                    grid-column: 1 / -1;
                }
                
                .chart-container h3 {
                    margin-top: 0;
                    color: #333;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 10px;
                }
            </style>
        `;
    }
    
    async _createSpendingTrends(transactions, containerId) {
        const monthlyData = this._aggregateByMonth(transactions);
        
        // Create simple line chart using Canvas
        const container = document.getElementById(containerId);
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 200;
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        this._drawLineChart(ctx, monthlyData, {
            width: 400,
            height: 200,
            title: 'Monthly Spending',
            color: '#007bff'
        });
    }
    
    async _createCategoryDistribution(transactions, containerId) {
        const categoryData = this._aggregateByCategory(transactions);
        
        const container = document.getElementById(containerId);
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 300;
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        this._drawPieChart(ctx, categoryData, {
            width: 300,
            height: 300,
            title: 'Category Distribution'
        });
    }
    
    async _createAmountDistribution(transactions, containerId) {
        const amounts = transactions.map(t => Math.abs(t.amount));
        const histogram = this._createHistogram(amounts, 20);
        
        const container = document.getElementById(containerId);
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 200;
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        this._drawBarChart(ctx, histogram, {
            width: 400,
            height: 200,
            title: 'Amount Distribution',
            color: '#28a745'
        });
    }
    
    async _createDuplicateAnalysis(transactions, containerId) {
        const duplicateStats = this._calculateDuplicateStats(transactions);
        
        const container = document.getElementById(containerId);
        const statsDiv = document.createElement('div');
        statsDiv.innerHTML = `
            <div class="duplicate-stats">
                <div class="stat-row">
                    <span>Total Transactions:</span>
                    <span class="stat-value">${duplicateStats.total}</span>
                </div>
                <div class="stat-row">
                    <span>Unique Transactions:</span>
                    <span class="stat-value">${duplicateStats.unique}</span>
                </div>
                <div class="stat-row">
                    <span>Duplicate Transactions:</span>
                    <span class="stat-value">${duplicateStats.duplicates}</span>
                </div>
                <div class="stat-row">
                    <span>Duplicate Groups:</span>
                    <span class="stat-value">${duplicateStats.groups}</span>
                </div>
            </div>
            <style>
                .duplicate-stats {
                    padding: 20px;
                }
                .stat-row {
                    display: flex;
                    justify-content: space-between;
                    margin: 10px 0;
                    padding: 8px;
                    background: #f8f9fa;
                    border-radius: 4px;
                }
                .stat-row .stat-value {
                    font-weight: bold;
                    color: #007bff;
                }
            </style>
        `;
        container.appendChild(statsDiv);
    }
    
    async _createConfidenceAnalysis(transactions, containerId) {
        const confidenceData = transactions.map(t => ({
            category: t.category || 'Uncategorized',
            confidence: t.confidence
        }));
        
        const avgByCategory = _.groupBy(confidenceData, 'category');
        const averages = Object.entries(avgByCategory).map(([category, items]) => ({
            category,
            avgConfidence: _.meanBy(items, 'confidence')
        }));
        
        const container = document.getElementById(containerId);
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 200;
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        this._drawBarChart(ctx, averages, {
            width: 400,
            height: 200,
            title: 'Confidence by Category',
            color: '#ffc107',
            valueKey: 'avgConfidence',
            labelKey: 'category'
        });
    }
    
    async _createTimelineView(transactions, containerId) {
        const container = document.getElementById(containerId);
        
        // Create scrollable timeline
        const timeline = document.createElement('div');
        timeline.className = 'timeline-container';
        timeline.innerHTML = `
            <div class="timeline-scroll">
                ${transactions
                    .sort((a, b) => new Date(b.date) - new Date(a.date))
                    .slice(0, 50) // Show latest 50 transactions
                    .map(t => `
                        <div class="timeline-item ${t.isDuplicate ? 'duplicate' : ''}">
                            <div class="timeline-date">${new Date(t.date).toLocaleDateString()}</div>
                            <div class="timeline-amount">$${Math.abs(t.amount).toFixed(2)}</div>
                            <div class="timeline-description">${t.description}</div>
                            <div class="timeline-category">${t.category || 'Uncategorized'}</div>
                            ${t.isDuplicate ? '<div class="duplicate-badge">DUPLICATE</div>' : ''}
                        </div>
                    `).join('')}
            </div>
            <style>
                .timeline-container {
                    max-height: 400px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .timeline-scroll {
                    padding: 10px;
                }
                .timeline-item {
                    display: grid;
                    grid-template-columns: 100px 80px 1fr 120px;
                    gap: 10px;
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                    position: relative;
                }
                .timeline-item.duplicate {
                    background-color: #fff3cd;
                    border-left: 3px solid #ffc107;
                }
                .timeline-date {
                    font-size: 12px;
                    color: #666;
                }
                .timeline-amount {
                    font-weight: bold;
                    color: #28a745;
                }
                .timeline-description {
                    color: #333;
                }
                .timeline-category {
                    font-size: 11px;
                    background: #007bff;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 10px;
                    text-align: center;
                }
                .duplicate-badge {
                    position: absolute;
                    top: 5px;
                    right: 5px;
                    background: #dc3545;
                    color: white;
                    font-size: 10px;
                    padding: 2px 6px;
                    border-radius: 3px;
                }
            </style>
        `;
        
        container.appendChild(timeline);
    }
    
    _updateSummaryStats(transactions) {
        const container = document.getElementById('summary-stats');
        
        const totalAmount = transactions.reduce((sum, t) => sum + Math.abs(t.amount), 0);
        const duplicateCount = transactions.filter(t => t.isDuplicate).length;
        const categories = new Set(transactions.map(t => t.category)).size;
        const avgConfidence = transactions.reduce((sum, t) => sum + t.confidence, 0) / transactions.length;
        
        container.innerHTML = `
            <div class="stat-item">
                <div class="stat-value">${transactions.length}</div>
                <div class="stat-label">Total Transactions</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">$${totalAmount.toLocaleString()}</div>
                <div class="stat-label">Total Amount</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${duplicateCount}</div>
                <div class="stat-label">Duplicates Found</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${categories}</div>
                <div class="stat-label">Categories</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${(avgConfidence * 100).toFixed(1)}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        `;
    }
    
    // Simple chart drawing methods
    _drawLineChart(ctx, data, options) {
        const { width, height, color, title } = options;
        const padding = 40;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw axes
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        if (data.length === 0) return;
        
        // Draw data
        const maxValue = Math.max(...data.map(d => d.value));
        const xStep = chartWidth / (data.length - 1);
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        data.forEach((point, index) => {
            const x = padding + index * xStep;
            const y = height - padding - (point.value / maxValue) * chartHeight;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
    }
    
    _drawPieChart(ctx, data, options) {
        const { width, height } = options;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 2 - 20;
        
        const total = data.reduce((sum, d) => sum + d.value, 0);
        let currentAngle = 0;
        
        const colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14', '#20c997'];
        
        data.forEach((item, index) => {
            const sliceAngle = (item.value / total) * 2 * Math.PI;
            
            ctx.fillStyle = colors[index % colors.length];
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
            ctx.closePath();
            ctx.fill();
            
            currentAngle += sliceAngle;
        });
    }
    
    _drawBarChart(ctx, data, options) {
        const { width, height, color, valueKey = 'value', labelKey = 'label' } = options;
        const padding = 40;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        ctx.clearRect(0, 0, width, height);
        
        if (data.length === 0) return;
        
        const maxValue = Math.max(...data.map(d => d[valueKey] || d.value));
        const barWidth = chartWidth / data.length * 0.8;
        const barSpacing = chartWidth / data.length * 0.2;
        
        data.forEach((item, index) => {
            const value = item[valueKey] || item.value;
            const barHeight = (value / maxValue) * chartHeight;
            const x = padding + index * (barWidth + barSpacing);
            const y = height - padding - barHeight;
            
            ctx.fillStyle = color;
            ctx.fillRect(x, y, barWidth, barHeight);
        });
    }
    
    // Helper methods for data aggregation
    _aggregateByMonth(transactions) {
        const monthlyMap = new Map();
        
        transactions.forEach(transaction => {
            const monthKey = `${transaction.date.getFullYear()}-${String(transaction.date.getMonth() + 1).padStart(2, '0')}`;
            
            if (!monthlyMap.has(monthKey)) {
                monthlyMap.set(monthKey, { label: monthKey, value: 0 });
            }
            
            monthlyMap.get(monthKey).value += Math.abs(transaction.amount);
        });
        
        return Array.from(monthlyMap.values()).sort((a, b) => a.label.localeCompare(b.label));
    }
    
    _aggregateByCategory(transactions) {
        const categoryMap = new Map();
        
        transactions.forEach(transaction => {
            const category = transaction.category || 'Uncategorized';
            
            if (!categoryMap.has(category)) {
                categoryMap.set(category, { label: category, value: 0 });
            }
            
            categoryMap.get(category).value += Math.abs(transaction.amount);
        });
        
        return Array.from(categoryMap.values());
    }
    
    _createHistogram(data, bins) {
        const min = Math.min(...data);
        const max = Math.max(...data);
        const binSize = (max - min) / bins;
        
        const histogram = Array(bins).fill(0).map((_, i) => ({
            label: `${(min + i * binSize).toFixed(0)}-${(min + (i + 1) * binSize).toFixed(0)}`,
            value: 0
        }));
        
        data.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
            histogram[binIndex].value++;
        });
        
        return histogram;
    }
    
    _calculateDuplicateStats(transactions) {
        const duplicates = transactions.filter(t => t.isDuplicate);
        const groups = new Set(duplicates.map(t => t.duplicateGroup)).size;
        
        return {
            total: transactions.length,
            unique: transactions.length - duplicates.length,
            duplicates: duplicates.length,
            groups: groups
        };
    }
}

/**
 * Main financial processing pipeline orchestrator
 */
class FinancialProcessor {
    constructor(config = {}) {
        this.config = config instanceof ProcessingConfig ? config : new ProcessingConfig(config);
        this.logger = console;
        
        // Initialize components
        this.dataLoader = new DataLoader(this.config);
        this.categorizer = new TransactionCategorizer(this.config);
        this.duplicateDetector = new DuplicateDetector(this.config);
        this.visualizer = new FinancialVisualizer(this.config);
        
        this.logger.info('Financial processor initialized');
    }
    
    async processFile(file) {
        this.logger.info('Starting financial data processing pipeline...');
        const startTime = performance.now();
        
        try {
            // Load transactions from file
            const transactions = await this.dataLoader.loadWorkbook(file);
            
            // Categorize transactions
            const categorized = await this.categorizer.categorizeTransactions(transactions);
            
            // Detect duplicates
            const processed = await this.duplicateDetector.detectDuplicates(categorized);
            
            // Generate summary
            const summary = this._generateSummary(processed);
            
            const processingTime = performance.now() - startTime;
            this.logger.info(`Processing completed in ${processingTime.toFixed(2)}ms`);
            
            return {
                transactions: processed,
                summary,
                processingTime,
                config: this.config
            };
            
        } catch (error) {
            this.logger.error('Error during processing:', error);
            throw error;
        }
    }
    
    async generateDashboard(results, containerId = 'dashboard') {
        this.logger.info('Generating interactive dashboard...');
        
        await this.visualizer.createComprehensiveDashboard(
            results.transactions, 
            containerId
        );
        
        this.logger.info('Dashboard generated successfully');
    }
    
    async exportResults(results, format = 'json') {
        this.logger.info(`Exporting results in ${format} format...`);
        
        switch (format.toLowerCase()) {
            case 'json':
                return this._exportJSON(results);
            case 'csv':
                return this._exportCSV(results);
            case 'xlsx':
                return this._exportExcel(results);
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    }
    
    _generateSummary(transactions) {
        const duplicates = transactions.filter(t => t.isDuplicate);
        const categories = _.groupBy(transactions, 'category');
        const amounts = transactions.map(t => t.amount);
        const confidences = transactions.map(t => t.confidence);
        
        return {
            totalTransactions: transactions.length,
            uniqueTransactions: transactions.length - duplicates.length,
            duplicateCount: duplicates.length,
            totalAmount: _.sum(amounts.map(Math.abs)),
            dateRange: {
                start: _.minBy(transactions, 'date').date,
                end: _.maxBy(transactions, 'date').date
            },
            categories: {
                count: Object.keys(categories).length,
                distribution: _.mapValues(categories, group => 
                    _.sum(group.map(t => Math.abs(t.amount)))
                )
            },
            confidenceStats: {
                mean: _.mean(confidences),
                median: this._calculateMedian(confidences),
                std: this._calculateStandardDeviation(confidences)
            }
        };
    }
    
    _exportJSON(results) {
        return JSON.stringify({
            summary: results.summary,
            transactions: results.transactions.map(t => t.toJSON()),
            processingTime: results.processingTime,
            exportTime: new Date().toISOString()
        }, null, 2);
    }
    
    _exportCSV(results) {
        const headers = [
            'Date', 'Amount', 'Description', 'Category', 'Confidence',
            'Is_Duplicate', 'Duplicate_Group', 'Account', 'Source_Sheet'
        ];
        
        const rows = results.transactions.map(t => [
            t.date.toISOString().split('T')[0],
            t.amount,
            `"${t.description.replace(/"/g, '""')}"`,
            t.category || '',
            t.confidence,
            t.isDuplicate,
            t.duplicateGroup || '',
            t.account,
            t.sourceSheet
        ]);
        
        return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    }
    
    async _exportExcel(results) {
        // Create workbook with processed data
        const wb = XLSX.utils.book_new();
        
        // Main transactions sheet
        const transactionData = results.transactions.map(t => ({
            Date: t.date,
            Amount: t.amount,
            Description: t.description,
            Category: t.category,
            Confidence: t.confidence,
            Is_Duplicate: t.isDuplicate,
            Duplicate_Group: t.duplicateGroup,
            Account: t.account,
            Source_Sheet: t.sourceSheet
        }));
        
        const ws = XLSX.utils.json_to_sheet(transactionData);
        XLSX.utils.book_append_sheet(wb, ws, 'Processed_Transactions');
        
        // Summary sheet
        const summaryData = [
            ['Metric', 'Value'],
            ['Total Transactions', results.summary.totalTransactions],
            ['Unique Transactions', results.summary.uniqueTransactions],
            ['Duplicate Count', results.summary.duplicateCount],
            ['Total Amount', results.summary.totalAmount],
            ['Categories Found', results.summary.categories.count],
            ['Average Confidence', results.summary.confidenceStats.mean]
        ];
        
        const summaryWs = XLSX.utils.aoa_to_sheet(summaryData);
        XLSX.utils.book_append_sheet(wb, summaryWs, 'Summary');
        
        return XLSX.write(wb, { type: 'binary', bookType: 'xlsx' });
    }
    
    _calculateMedian(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 
            ? (sorted[mid - 1] + sorted[mid]) / 2 
            : sorted[mid];
    }
    
    _calculateStandardDeviation(values) {
        const mean = _.mean(values);
        const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
        return Math.sqrt(_.mean(squaredDiffs));
    }
}

// Export for use in web applications
export { 
    FinancialProcessor, 
    ProcessingConfig, 
    TransactionRecord,
    DataLoader,
    TransactionCategorizer,
    DuplicateDetector,
    FinancialVisualizer
};

// Example usage for web environment
if (typeof window !== 'undefined') {
    window.FinancialProcessor = FinancialProcessor;
    window.ProcessingConfig = ProcessingConfig;
    
    // DOM ready initialization
    document.addEventListener('DOMContentLoaded', () => {
        console.log('Financial Transaction Processor loaded and ready');
        
        // Example initialization
        const processor = new FinancialProcessor();
        
        // Add file input handler if present
        const fileInput = document.getElementById('financial-file-input');
        if (fileInput) {
            fileInput.addEventListener('change', async (event) => {
                const file = event.target.files[0];
                if (file) {
                    try {
                        const results = await processor.processFile(file);
                        await processor.generateDashboard(results);
                        console.log('Processing completed:', results.summary);
                    } catch (error) {
                        console.error('Processing failed:', error);
                    }
                }
            });
        }
    });
}
