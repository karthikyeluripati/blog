const PublicationPage = () => {
    const publications = [
        {
            id: 1,
            title: "Forecasting Electrical Demand for the Residential Sector at the National Level Using Deep Learning",
            authors: ["P. K. Dharmoju*", "K. Yeluripati*", "J. Guduri*", "K. Palle"],
            venue: "International Conference on Artificial Intelligence and Machine Vision (2021)",
            image: "images/research/mp_l.png", 
            link: "https://ieeexplore.ieee.org/document/9670956"
          },
          // {
          //   id: 1,
          //   title: "Forecasting Electrical Demand for the Residential Sector at the National Level Using Deep Learning",
          //   authors: ["P. K. Dharmoju*", "K. Yeluripati*", "J. Guduri*", "K. Palle"],
          //   venue: "International Conference on Artificial Intelligence and Machine Vision (2021)",
          //   image: "images/research/mp_l.png", 
          //   caption: "\"put the toothpaste in the wooden bowl\"",
          //   link: "https://ieeexplore.ieee.org/document/9670956"
          // }
    ];
   
    const formatAuthors = (authors: any[]) => {
      return authors.map((author, index) => {
        const isLastAuthor = index === authors.length - 1;
        const separator = isLastAuthor ? "" : ", ";
        
        // Change 'Your Name' to the name you want to bold
        if (author === "K. Yeluripati*") {
          return (
            <span key={index}>
              <strong className="font-semibold">{author}</strong>{separator}
            </span>
          );
        }
        
        return <span key={index}>{author}{separator}</span>;
      });
    };
  
    return (
      <div className="max-w-4xl mx-auto">
        
        <div className="space-y-12">
          {publications.map(pub => (
            <div 
              key={pub.id}
              className="group flex gap-6 hover:bg-gray-50 p-4 -mx-4 rounded-lg transition-colors"
            >
              <div className="w-48 flex-shrink-0">
                <img
                  src={pub.image}
                  alt={pub.title}
                  className="rounded-lg w-full object-cover"
                />
                {/* {pub.caption && (
                  <p className="text-sm text-gray-600 mt-2 italic text-center">
                    {pub.caption}
                  </p>
                )} */}
              </div>
  
              <div className="flex-1 space-y-2">
                <a 
                  href={pub.link}
                  className="text-black hover:text-blue-700 text-xl font-medium group-hover:"
                >
                  {pub.title}
                </a>
                <p className="text-gray-800">{formatAuthors(pub.authors)}</p>
                <p className="text-gray-600 italic">{pub.venue}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };
  
  export default PublicationPage;