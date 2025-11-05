# 🔐 Role-Based Access Control

## Overview
AeroVision implements role-based access control (RBAC) to ensure users only access features relevant to their responsibilities.

## 👥 User Roles

### 👑 Admin (Full Access)
**Username:** `admin`  
**Password:** `admin123`

**Access Level:** Complete system access

**Available Features:**
- ✅ All Core Features (Dashboard, Flight Map, 3D Visualization)
- ✅ All Operations (Predict, Batch Prediction, Advanced Search)
- ✅ All Analytics (Analytics, Historical Compare, Reports, Data Quality)
- ✅ All Management (Alerts, Weather, Collaboration, Scenario Simulator)
- ✅ All System Features (API Integration, Help & Guide, System Status)

**Responsibilities:**
- System configuration
- User management
- API integration
- Full operational control

---

### ✈️ Controller (Operations Focus)
**Username:** `controller`  
**Password:** `atc123`

**Access Level:** Operational features

**Available Features:**
- ✅ Dashboard - Real-time flight monitoring
- ✅ Flight Map - Live flight tracking
- ✅ 3D Visualization - Interactive flight paths
- ✅ Predict Delay - Single flight predictions
- ✅ Batch Prediction - Multiple flight predictions
- ✅ Advanced Search - Flight lookup and filtering
- ✅ Alerts - Delay notifications
- ✅ Weather - Weather impact analysis
- ✅ Collaboration - Team communication
- ✅ Help & Guide - Documentation

**Restricted From:**
- ❌ Analytics & Reports
- ❌ Data Quality Dashboard
- ❌ Historical Comparison
- ❌ Scenario Simulator
- ❌ API Integration
- ❌ System Status

**Responsibilities:**
- Monitor real-time flight operations
- Make delay predictions
- Manage alerts and notifications
- Coordinate with team members

---

### 📊 Analyst (Analytics Focus)
**Username:** `analyst`  
**Password:** `analyst123`

**Access Level:** Analytics and reporting

**Available Features:**
- ✅ Dashboard - Overview statistics
- ✅ Flight Map - Geographic analysis
- ✅ 3D Visualization - Data visualization
- ✅ Analytics - Performance metrics
- ✅ Historical Compare - Trend analysis
- ✅ Reports - Generate and export reports
- ✅ Data Quality - Data validation
- ✅ Advanced Search - Data exploration
- ✅ Help & Guide - Documentation

**Restricted From:**
- ❌ Predict Delay
- ❌ Batch Prediction
- ❌ Alerts Management
- ❌ Weather Integration
- ❌ Collaboration Tools
- ❌ Scenario Simulator
- ❌ API Integration
- ❌ System Status

**Responsibilities:**
- Analyze flight performance
- Generate reports
- Monitor data quality
- Identify trends and patterns

---

## 🔒 Security Features

### Authentication
- Secure password hashing (SHA-256)
- Session-based authentication
- Automatic logout on session end

### Authorization
- Page-level access control
- Role-based feature filtering
- Permission checks on every page load

### Visual Indicators
- Color-coded role badges
- Role-specific icons
- Clear access level display

---

## 🎨 Role Visual Identity

### Admin
- **Color:** Green (#00ff88)
- **Icon:** 👑 Crown
- **Theme:** Authority and full control

### Controller
- **Color:** Blue (#00aaff)
- **Icon:** ✈️ Airplane
- **Theme:** Operations and real-time action

### Analyst
- **Color:** Purple (#bb86fc)
- **Icon:** 📊 Chart
- **Theme:** Data and insights

---

## 📋 Feature Access Matrix

| Feature | Admin | Controller | Analyst |
|---------|-------|------------|---------|
| Dashboard | ✅ | ✅ | ✅ |
| Flight Map | ✅ | ✅ | ✅ |
| 3D Visualization | ✅ | ✅ | ✅ |
| Predict Delay | ✅ | ✅ | ❌ |
| Batch Prediction | ✅ | ✅ | ❌ |
| Advanced Search | ✅ | ✅ | ✅ |
| Analytics | ✅ | ❌ | ✅ |
| Historical Compare | ✅ | ❌ | ✅ |
| Reports | ✅ | ❌ | ✅ |
| Data Quality | ✅ | ❌ | ✅ |
| Alerts | ✅ | ✅ | ❌ |
| Weather | ✅ | ✅ | ❌ |
| Collaboration | ✅ | ✅ | ❌ |
| Scenario Simulator | ✅ | ❌ | ❌ |
| API Integration | ✅ | ❌ | ❌ |
| System Status | ✅ | ❌ | ❌ |
| Help & Guide | ✅ | ✅ | ✅ |

---

## 🚀 Testing Role-Based Access

### Test Admin Access
1. Login with `admin` / `admin123`
2. Verify all navigation options are visible
3. Access any page - all should work

### Test Controller Access
1. Login with `controller` / `atc123`
2. Verify only operational pages are visible
3. Try accessing Analytics - should not be in navigation
4. Verify predictions and alerts work

### Test Analyst Access
1. Login with `analyst` / `analyst123`
2. Verify only analytics pages are visible
3. Try accessing Predict Delay - should not be in navigation
4. Verify reports and data quality work

---

## 🔧 Customizing Roles

To add or modify roles, edit the `has_permission()` function in `app.py`:

```python
role_permissions = {
    'your_role': [
        '📊 Dashboard',
        '🗺️ Flight Map',
        # Add allowed pages here
    ]
}
```

---

## 📞 Support

If you experience access issues:
1. Verify you're using the correct credentials
2. Check your role assignment
3. Clear browser cache and re-login
4. Contact system administrator

---

**Note:** All passwords are hashed using SHA-256 for security. Never share your credentials.
