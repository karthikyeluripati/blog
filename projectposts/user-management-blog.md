---
title: "Scalable User Management System"
subtitle: "Modular, secure, and scalable solution using Node.js, Express and MongoDB"
date: "02-2024"
---

<p align="center">[GitHub](https://github.com/karthikyeluripati/user_management)</p>

This project delves into the intricacies of building a scalable and secure User Management System using Node.js, Express, JWT authentication, and MongoDB. We'll explore the architecture, key features, and technical challenges faced during development.(Note- attached link refers to code in python)

## Project Overview

The User Management System is designed to be a modular, secure, and scalable solution for handling user-related operations in web applications. It encompasses critical functionalities such as user registration, authentication, role-based access control (RBAC), and profile management.

### Key Features

1. User Registration and Authentication
2. Role-Based Access Control (RBAC)
3. User Profile Management
4. Scalable Database Design

## Technical Stack

- **Backend**: Node.js with Express framework
- **Database**: MongoDB with Mongoose ORM
- **Authentication**: JSON Web Tokens (JWT)
- **Password Hashing**: bcrypt
- **Frontend**: HTML, CSS, and JavaScript (not the focus of this blog)

## Project Structure

The project follows a modular structure to ensure separation of concerns:

```
.
├── controllers
│   └── userController.js
├── middleware
│   └── authMiddleware.js
├── models
│   └── User.js
├── routes
│   └── userRoutes.js
├── config
│   └── db.js
├── server.js
└── package.json
```

This structure allows for easy maintenance and scalability as the project grows.

## User Authentication and Authorization

### JWT-based Authentication

JSON Web Tokens (JWT) provide a stateless and secure method for user authentication. Let's break down the implementation:

#### 1. User Registration

```javascript
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');

exports.registerUser = async (req, res) => {
  const { name, email, password, role } = req.body;

  // Hash the password
  const salt = await bcrypt.genSalt(10);
  const hashedPassword = await bcrypt.hash(password, salt);

  // Create user
  const user = new User({
    name,
    email,
    password: hashedPassword,
    role
  });

  await user.save();

  // Generate JWT token
  const token = jwt.sign(
    { id: user._id, role: user.role },
    process.env.JWT_SECRET,
    { expiresIn: '1h' }
  );

  res.status(201).json({ token });
};
```

This code snippet demonstrates:
- Password hashing using bcrypt for security
- User creation in MongoDB
- JWT token generation with user ID and role

#### 2. User Login

```javascript
exports.loginUser = async (req, res) => {
  const { email, password } = req.body;

  // Check for user
  const user = await User.findOne({ email });
  if (!user) return res.status(400).json({ message: 'User not found' });

  // Validate password
  const isMatch = await bcrypt.compare(password, user.password);
  if (!isMatch) return res.status(400).json({ message: 'Invalid credentials' });

  // Generate JWT token
  const token = jwt.sign(
    { id: user._id, role: user.role },
    process.env.JWT_SECRET,
    { expiresIn: '1h' }
  );

  res.status(200).json({ token });
};
```

This login process ensures:
- User existence check
- Secure password comparison
- JWT token generation upon successful authentication

### Role-Based Access Control (RBAC)

RBAC is implemented using middleware to protect routes based on user roles:

```javascript
exports.protect = (roles = []) => {
  return (req, res, next) => {
    const token = req.headers.authorization.split(' ')[1];
    if (!token) return res.status(401).json({ message: 'Not authorized' });

    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      req.user = decoded;

      // Role-based access
      if (!roles.includes(req.user.role)) {
        return res.status(403).json({ message: 'Forbidden: You don't have permission' });
      }

      next();
    } catch (error) {
      return res.status(401).json({ message: 'Token is invalid' });
    }
  };
};
```

This middleware:
- Verifies the JWT token
- Checks user roles against allowed roles for the route
- Allows or denies access based on role

Usage in routes:

```javascript
const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/authMiddleware');
const { getUserProfile } = require('../controllers/userController');

// Route only accessible by admins
router.get('/admin', protect(['admin']), (req, res) => {
  res.json({ message: 'Welcome, Admin!' });
});

// User profile (accessible by all authenticated users)
router.get('/profile', protect(['user', 'admin']), getUserProfile);
```

## Database Design with MongoDB

MongoDB's flexible schema design is leveraged for efficient user data storage. Here's the User schema:

```javascript
const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
  },
  password: {
    type: String,
    required: true,
  },
  role: {
    type: String,
    enum: ['admin', 'user', 'viewer'],
    default: 'user',
  },
}, {
  timestamps: true
});

module.exports = mongoose.model('User', userSchema);
```

This schema design:
- Ensures email uniqueness
- Implements role-based access control through the `role` field
- Includes timestamps for auditing purposes

## Error Handling and Security Measures

### Input Validation

To prevent injection attacks and ensure data integrity, input validation is implemented using Express Validator:

```javascript
const { check, validationResult } = require('express-validator');

exports.validateRegister = [
  check('email', 'Email is required').isEmail(),
  check('password', 'Password must be 6 or more characters').isLength({ min: 6 }),
  (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) return res.status(400).json({ errors: errors.array() });
    next();
  },
];
```

### Additional Security Measures

1. **Password Hashing**: bcrypt is used to securely hash passwords before storage.
2. **JWT Token Expiry**: Tokens are set to expire after a specified time to mitigate risks associated with token theft.
3. **Environment Variables**: Sensitive data like the JWT secret key is stored in `.env` files, keeping it out of the codebase.

## Challenges and Learnings

1. **Session Management**: Implementing secure token refresh mechanisms was challenging but crucial for maintaining user sessions without compromising security.

2. **Fine-grained RBAC**: Designing a flexible yet secure role-based access control system required careful consideration to prevent privilege escalation.

## Future Improvements

1. **OAuth2 Integration**: Implementing third-party authentication providers like Google or GitHub.
2. **Two-Factor Authentication (2FA)**: Enhancing security, especially for admin accounts.
3. **Advanced User Analytics**: Implementing logging and analytics to track user behavior and system usage.

## Conclusion

Building this User Management System provided invaluable insights into modern authentication practices, secure API design, and scalable database management. The combination of Node.js, Express, JWT, and MongoDB proved to be a powerful stack for creating a robust, secure, and scalable user management solution.
