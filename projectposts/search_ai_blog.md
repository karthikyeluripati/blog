---
title: "SearchAI"
subtitle: "buildspace s5 - A Feature-Rich PyQt5 Browser with Integrated AI"
date: "08-2024"
producturl: "https://github.com/karthikyeluripati/searchai"
media:
  - type: "image"
    url: "/images/searchai/buildspace.webp"
    link: "/images/searchai/buildspace.webp"
  - type: "video"
    url: "/videos/searchai/2024-07-01 23-35-38.mkv"
    link: "https://www.youtube.com/watch?v=e6KWyaee2Ls"
    # thumbnail: "/images/projects/project1/video-thumbnail.jpg"
  - type: "image"
    url: "/images/searchai/image.png"
    link: "/images/searchai/image.png"
---

<p align='center'>[GitHub](https://github.com/karthikyeluripati/searchai) [Tweet](https://x.com/Karthik_Ysvk/status/1808008945232949443) [Youtube](https://www.youtube.com/watch?v=e6KWyaee2Ls)</p>

This project showcases a custom web browser built from scratch using Python and PyQt5, with integrated AI features. The browser combines traditional web browsing capabilities with modern AI-powered functionalities, creating a unique and powerful user experience.

![Buildspace s5 poster](/images/searchai/image.png)

## Core Technology Stack

- **Python**: The primary programming language used for the project.
- **PyQt5**: A comprehensive GUI framework for creating desktop applications.
- **QtWebEngine**: Provides web browsing capabilities within the PyQt5 application.
- **Custom AI Integration**: Implemented through the `QuickAgent` module (not shown in the provided code).

## Key Features

1. Traditional Web Browsing
2. Voice Assistant
3. Smart To-Do List
4. Calendar Management
5. Tab Grouping
6. Theme Generator
7. Spotify Mood Music Integration
8. Screen Time Tracking

Let's explore each component in detail.

## Browser Architecture

The main class `FeatureBrowser` inherits from `QMainWindow`, serving as the core of the application. Here's a breakdown of its structure:

```python
class FeatureBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature-Rich Browser")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.create_toolbar()
        self.create_landing_page()
        self.create_web_view()
```

This setup creates a main window with a central widget and a vertical layout. The browser is composed of three main parts:

1. Toolbar
2. Landing Page
3. Web View

### 1. Toolbar

The toolbar provides basic navigation functions:

```python
def create_toolbar(self):
    navbar = QToolBar()
    self.addToolBar(navbar)

    back_btn = QPushButton("Back")
    back_btn.clicked.connect(self.go_back)
    navbar.addWidget(back_btn)

    # ... (forward, reload, and home buttons)

    self.url_bar = QLineEdit()
    self.url_bar.returnPressed.connect(self.navigate_to_url)
    navbar.addWidget(self.url_bar)
```

This code creates a toolbar with back, forward, reload, and home buttons, as well as a URL bar for direct navigation.

### 2. Landing Page

The landing page is where most of the AI-powered features are presented:

```python
def create_landing_page(self):
    self.landing_page = QWidget()
    landing_layout = QVBoxLayout(self.landing_page)

    # Voice Assistant
    voice_layout = QHBoxLayout()
    voice_label = QLabel("Voice Assistant:")
    voice_button = QPushButton("Activate")
    voice_button.clicked.connect(lambda: click_voicebot(self))
    voice_layout.addWidget(voice_label)
    voice_layout.addWidget(voice_button)
    landing_layout.addLayout(voice_layout)

    # ... (other features like To-Do List, Calendar, etc.)
```

Each feature is implemented as a widget or set of widgets, arranged in the landing page layout.

### 3. Web View

The web view uses `QWebEngineView` to display web pages:

```python
def create_web_view(self):
    self.browser = QWebEngineView()
    self.browser.urlChanged.connect(self.update_url)
    self.layout.addWidget(self.browser)
    self.browser.hide()
```

The web view is initially hidden and shown when a URL is entered.

## AI-Powered Features

### Voice Assistant

The voice assistant is activated through a button click:

```python
def click_voicebot(browser):
    manager = ConversationManager(browser)
    asyncio.run(manager.main())
```

This function creates a `ConversationManager` instance and runs its main method asynchronously. The `ConversationManager` class is likely defined in the `QuickAgent` module, which handles the AI-powered voice interactions.

### Smart To-Do List

The to-do list is implemented using a `QListWidget`:

```python
todo_layout = QVBoxLayout()
todo_label = QLabel("Smart To-Do List:")
todo_list = QListWidget()
todo_layout.addWidget(todo_label)
todo_layout.addWidget(todo_list)
```

While the current implementation doesn't show the "smart" features, it provides a foundation for adding AI-powered task management, such as automatic prioritization or context-based suggestions.

### Calendar Management

A `QCalendarWidget` is used for calendar functionality:

```python
calendar_label = QLabel("Calendar Management:")
calendar = QCalendarWidget()
landing_layout.addWidget(calendar_label)
landing_layout.addWidget(calendar)
```

This basic calendar could be enhanced with AI features like smart event scheduling or conflict resolution.

### Tab Grouping

Tab grouping is implemented with a color selection dropdown:

```python
tab_layout = QHBoxLayout()
tab_label = QLabel("Tab Grouping:")
tab_color = QComboBox()
tab_color.addItems(["Red", "Green", "Blue", "Yellow"])
tab_layout.addWidget(tab_label)
tab_layout.addWidget(tab_color)
```

This feature could be expanded to use AI for automatic tab categorization based on content.

### Theme Generator

A button is provided for theme generation:

```python
theme_layout = QHBoxLayout()
theme_label = QLabel("Theme Generator:")
theme_button = QPushButton("Generate Theme")
theme_layout.addWidget(theme_label)
theme_layout.addWidget(theme_button)
```

The AI could generate custom themes based on user preferences or browsing history.

### Spotify Mood Music

A slider is used to represent mood for music selection:

```python
spotify_layout = QVBoxLayout()
spotify_label = QLabel("Spotify Mood Music:")
mood_slider = QSlider(Qt.Horizontal)
mood_slider.setRange(0, 100)
spotify_layout.addWidget(spotify_label)
spotify_layout.addWidget(mood_slider)
```

This feature could use AI to analyze the user's mood based on browsing behavior and adjust music recommendations accordingly.

### Screen Time Tracking

A simple label is used to display screen time:

```python
screen_time_label = QLabel("Screen Time: 0h 0m")
landing_layout.addWidget(screen_time_label)
```

AI could be employed to provide insights on usage patterns and suggest healthy browsing habits.

## Conclusion

AI Web Browser project demonstrates an innovative approach to combining traditional web browsing with AI-powered features. The PyQt5 framework provides a solid foundation for building a custom browser, while the integration of AI technologies opens up numerous possibilities for enhancing the user experience.

To further develop this project, consider the following suggestions:

1. Implement the AI logic for each feature, possibly using machine learning models or natural language processing.
2. Enhance the voice assistant to perform web searches and control browser functions.
3. Develop the smart to-do list to automatically categorize and prioritize tasks.
4. Integrate the calendar with AI for smart scheduling and reminders.
5. Implement AI-driven tab grouping based on content analysis.
6. Create an AI theme generator that considers user preferences and current trends.
7. Develop the Spotify integration to recommend music based on browsing mood and history.
8. Enhance screen time tracking with AI-powered insights and recommendations.

This project showcases the potential for AI to revolutionize web browsing, creating a more intuitive and personalized experience for users. As to continue and refine these features, we push the boundaries of what's possible in modern web browsers.