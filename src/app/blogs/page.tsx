import getPostMetadata from '../../../components/getPostMetadata';
import PostPreview from '../../../components/PostPreview';

const HomePage = () => {
  const postMetadata = getPostMetadata();

  // Helper function to parse the date string
  const parseDate = (dateStr) => {
    const [day, month, year] = dateStr.split('-').map(Number);
    // Note: JavaScript months are 0-indexed, so we subtract 1 from the month
    return new Date(2000 + year, month - 1, day);
  };

  // Sort the postMetadata array by date in descending order (newest first)
  const sortedPostMetadata = postMetadata.sort((a, b) => {
    return parseDate(b.date) - parseDate(a.date);
  });

  const PostPreviews = sortedPostMetadata.map((post) => (
    <PostPreview key={post.slug} {...post} />
  ));

  return (
    <div className=''>
      <h1 className="text-2xl font-bold mb-4">Blogs</h1>
      {PostPreviews}
    </div>
  );
}

export default HomePage;