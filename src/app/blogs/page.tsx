
import  getPostMetadata  from '../../../components/getPostMetadata';
import PostPreview from '../../../components/PostPreview';

const HomePage = () => {
  const postMetadata = getPostMetadata();
  const PostPreviews = postMetadata.map((post)=>(
  <PostPreview key={post.slug} {...post} />
  ));
  // return <div className='grid grid-cols-1 md:grid-cols-2 gap-4'>{PostPreviews}</div>;
  return (
  <div className=''>
    <h1 className="text-2xl font-bold mb-4">Blogs</h1>
    {PostPreviews}
  </div>);
}

export default HomePage;